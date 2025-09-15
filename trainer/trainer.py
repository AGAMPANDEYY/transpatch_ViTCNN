import sys
# Save the original sys.path
original_sys_path = sys.path.copy()
sys.path.append("/kaggle/working/adversarial-patch-transferability/")
from dataset.cityscapes import Cityscapes

from pretrained_models.models import Models

from pretrained_models.ICNet.icnet import ICNet
from pretrained_models.BisNetV1.model import BiSeNetV1
from pretrained_models.BisNetV2.model import BiSeNetV2
from pretrained_models.PIDNet.model import PIDNet, get_pred_model

from metrics.performance import SegmentationMetric
from metrics.loss import PatchLoss
from patch.create import Patch
from torch.optim.lr_scheduler import ExponentialLR
import time
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt

# Restore original sys.path to avoid conflicts or shadowing
sys.path = original_sys_path

class PatchTrainer():
  def __init__(self,config,main_logger):
      self.config = config
      self.start_epoch = config.train.start_epoch
      self.end_epoch = config.train.end_epoch
      self.epochs = self.end_epoch - self.start_epoch
      self.batch_train = config.train.batch_size
      self.batch_test = config.test.batch_size
      self.device = config.experiment.device
      self.logger = main_logger
      self.lr = config.optimizer.init_lr
      self.power = config.train.power
      self.lr_scheduler = config.optimizer.exponentiallr
      self.lr_scheduler_gamma = config.optimizer.exponentiallr_gamma
      self.log_per_iters = config.train.log_per_iters
      self.patch_size = config.patch.size
      self.apply_patch = Patch(config).apply_patch
      self.epsilon = config.optimizer.init_lr

      cityscape_train = Cityscapes(
          root = config.dataset.root,
          list_path = config.dataset.train,
          num_classes = config.dataset.num_classes,
          multi_scale = config.train.multi_scale,
          flip = config.train.flip,
          ignore_label = config.train.ignore_label,
          base_size = config.train.base_size,
          crop_size = (config.train.height,config.train.width),
          scale_factor = config.train.scale_factor
        )

      cityscape_test = Cityscapes(
          root = config.dataset.root,
          list_path = config.dataset.val,
          num_classes = config.dataset.num_classes,
          multi_scale = False,
          flip = False,
          ignore_label = config.train.ignore_label,
          base_size = config.test.base_size,
          crop_size = (config.test.height,config.test.width),
        )
      
      self.train_dataloader = torch.utils.data.DataLoader(dataset=cityscape_train,
                                              batch_size=self.batch_train,
                                              shuffle=config.train.shuffle,
                                              num_workers=config.train.num_workers,
                                              pin_memory=config.train.pin_memory,
                                              drop_last=config.train.drop_last)
      self.test_dataloader = torch.utils.data.DataLoader(dataset=cityscape_test,
                                            batch_size=self.batch_test,
                                            shuffle=False,
                                            num_workers=config.test.num_workers,
                                            pin_memory=config.test.pin_memory,
                                            drop_last=config.test.drop_last)
      


      self.iters_per_epoch = len(self.train_dataloader)
      self.max_iters = self.end_epoch * self.iters_per_epoch

      ## Getting the model
      self.model = Models(self.config)
      self.model.get()

      ## loss
      self.criterion = PatchLoss(self.config)

      ## optimizer
      # Initialize adversarial patch (random noise)
      self.patch = torch.rand((3, self.patch_size, self.patch_size), 
                              requires_grad=True, 
                              device=self.device)
      
      # # Define optimizer
      # self.optimizer = torch.optim.SGD(params = [self.patch],
      #                         lr=self.lr,
      #                         momentum=config.optimizer.momentum,
      #                         weight_decay=config.optimizer.weight_decay,
      #                         nesterov=config.optimizer.nesterov,
      # )
      # if self.lr_scheduler:
      #   self.scheduler = ExponentialLR(self.optimizer, gamma=self.lr_scheduler_gamma)


      ## Initializing quantities
      self.metric = SegmentationMetric(config) 
      self.current_mIoU = 0.0
      self.best_mIoU = 0.0

      self.current_epoch = 0
      self.current_iteration = 0

  def train(self):
    epochs, iters_per_epoch, max_iters = self.epochs, self.iters_per_epoch, self.max_iters

    start_time = time.time()
    self.logger.info('Start training, Total Epochs: {:d} = Iterations per epoch {:d}'.format(epochs, iters_per_epoch))
    IoU = []
    for ep in range(self.start_epoch, self.end_epoch):
      self.current_epoch = ep
      self.metric.reset()
      total_loss = 0
      samplecnt = 0
      for i_iter, batch in enumerate(self.train_dataloader, 0):
          self.current_iteration += 1
          samplecnt += batch[0].shape[0]
          image, true_label,_, _, _ = batch
          image, true_label = image.to(self.device), true_label.to(self.device)
          

              
          
          # Randomly place patch in image and label(put ignore index)
          patched_image, patched_label = self.apply_patch(image,true_label,self.patch)
          # fig = plt.figure()
          # ax = fig.add_subplot(1,2,1)
          # ax.imshow(patched_image[0].permute(1,2,0).cpu().detach().numpy())
          # ax = fig.add_subplot(1,2,2)
          # ax.imshow(patched_label[0].cpu().detach().numpy())
          # plt.show()

          # Forward pass through the model (and interpolation if needed)
          output = self.model.predict(patched_image,patched_label.shape)
          #plt.imshow(output.argmax(dim =1)[0].cpu().detach().numpy())
          #plt.show()
          #break

          # Compute adaptive loss
          loss = self.criterion.compute_loss(output, patched_label)
          #loss = self.criterion.compute_loss_direct(output, patched_label)
          total_loss += loss.item()
          #break

          ## metrics
          self.metric.update(output, patched_label)
          pixAcc, mIoU = self.metric.get()

          # Backpropagation
          self.model.model.zero_grad()
          if self.patch.grad is not None:
            self.patch.grad.zero_()
          loss.backward()
          with torch.no_grad():
              #self.patch += self.epsilon * self.patch.grad.sign()  # Update patch using FGSM-style ascent
              self.patch += self.epsilon * self.patch.grad.data.sign()
              self.patch.clamp_(0, 1)  # Keep pixel values in valid range

          ## ETA
          eta_seconds = ((time.time() - start_time) / self.current_iteration) * (iters_per_epoch*epochs - self.current_iteration)
          eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

          if i_iter % self.log_per_iters == 0:
            self.logger.info(
              "Epochs: {:d}/{:d} || Samples: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || mIoU: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                  self.current_epoch, self.end_epoch,
                  samplecnt, self.batch_train*iters_per_epoch,
                  #self.optimizer.param_groups[0]['lr'],
                  self.epsilon,
                  loss.item(),
                  mIoU,
                  str(datetime.timedelta(seconds=int(time.time() - start_time))),
                  eta_string))
          

      average_pixAcc, average_mIoU = self.metric.get()
      average_loss = total_loss/len(self.train_dataloader)
      self.logger.info('-------------------------------------------------------------------------------------------------')
      self.logger.info("Epochs: {:d}/{:d}, Average loss: {:.3f}, Average mIoU: {:.3f}, Average pixAcc: {:.3f}".format(
        self.current_epoch, self.epochs, average_loss, average_mIoU, average_pixAcc))

      
      #self.test() ## Doing 1 iteration of testing
      self.logger.info('-------------------------------------------------------------------------------------------------')
      #self.model.train() ## Setting the model back to train mode
      # if self.lr_scheduler:
      #     self.scheduler.step()

      IoU.append(self.metric.get(full=True))

    return self.patch.detach(),np.array(IoU)  # Return adversarial patch and IoUs over epochs

    
