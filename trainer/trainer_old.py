import torch
import sys
sys.path.append('/content/drive/MyDrive/Colab Notebooks/1_Papers/2_RobustRealtimeSS/1_Pretraining_cityscape/1_PIDNet')
from dataset.cityscapes import Cityscapes
from model.model import get_seg_model
from metrics.loss import Pidnet_loss
from metrics.performance import SegmentationMetric
from torch.optim.lr_scheduler import ExponentialLR
import time
import datetime
import os
import datetime
import pytz

class Trainer():
    def __init__(self,config,main_logger):
        self.config = config
        self.start_epoch = config.train.start_epoch
        self.end_epoch = config.train.end_epoch
        self.epochs = self.end_epoch - self.start_epoch
        self.min_epoch = config.train.min_epoch
        self.max_epoch = config.train.max_epoch
        self.batch_train = config.train.batch_size
        self.batch_test = config.test.batch_size
        self.device = config.experiment.device
        self.logger = main_logger
        self.lr = config.optimizer.init_lr
        self.power = config.train.power
        self.lr_scheduler = config.optimizer.exponentiallr
        self.lr_scheduler_gamma = config.optimizer.exponentiallr_gamma
        self.log_per_iters = config.train.log_per_iters

        cityscape_train = Cityscapes(
          root = config.dataset.root,
          list_path = config.dataset.trainval,
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
        self.max_iters = self.max_epoch * self.iters_per_epoch

        #### create network
        self.model = get_seg_model(config,self.logger).to(self.device)


        # ## uploading a trained model 66.7% miou on Cityscape Val
        # wts = torch.load('/content/drive/MyDrive/Colab Notebooks/2_Pytorch/1_CNN/2_Segmentation/1_ICNet/experiments/run3_loss0_512x1024_hflip_polylr_mioucorrect/checkpoints/resnet50_2024-11-03 20:52:22 EST-0500_178_crop_512x1024_0.663.pth.tar', map_location=self.device)
        # self.model.load_state_dict(wts['model_state_dict'])



        # create criterion
        self.criterion = Pidnet_loss(
          config = config,
          weight = cityscape_train.class_weights
        ).to(self.device)


        # optimizer, for model just includes pretrained, head and auxlayer
        # params_list = list()
        # if hasattr(self.model, 'pretrained'):
        #     params_list.append({'params': self.model.pretrained.parameters(), 'lr': 0.01})

        # if hasattr(self.model, 'exclusive'):
        #   for module in self.model.exclusive:
        #       params_list.append({'params': getattr(self.model, module).parameters(), 'lr': cfg["optimizer"]["init_lr"] * 10})

        # self.optimizer = torch.optim.SGD(params = params_list,
        #                           lr = 0.01)

        self.optimizer = torch.optim.SGD(params = self.model.parameters(),
                                lr=self.lr,
                                momentum=config.optimizer.momentum,
                                weight_decay=config.optimizer.weight_decay,
                                nesterov=config.optimizer.nesterov,
                                )

        if self.lr_scheduler:
          self.scheduler = ExponentialLR(self.optimizer, gamma=self.lr_scheduler_gamma)
        
        # self.optimizer = torch.optim.SGD(params = self.model.parameters(), # will even optimize all mobileNet params
        #                                  lr = self.lr,
        #                                  momentum= config.optimizer.momentum,
        #                                  weight_decay= config.optimizer.weight_decay)



        # # dataparallel
        # if(self.dataparallel):
        #      self.model = nn.DataParallel(self.model)

        # evaluation metrics
        self.metric = SegmentationMetric(config) 
        self.current_mIoU = 0.0
        self.best_mIoU = 0.0

        self.current_epoch = 0
        self.current_iteration = 0

        if self.config.train.finetune:
          self.finetune_add = self.config.train.finetune_add
          ## Getting config and pretrained weights
          self.logger.info('Getting pretrained weights')
          wts = torch.load(self.finetune_add, map_location=config.experiment.device)

          ## Updating model and optimizer
          self.logger.info('Updating model and optimizer')
          self.model.load_state_dict(wts['model_state_dict'])
          self.optimizer.load_state_dict(wts['optimizer_state_dict'])
          if self.lr_scheduler:
            dummy_params = [torch.nn.Parameter(torch.zeros(1))]
            lrs = []
            opt = torch.optim.SGD(dummy_params, lr = self.lr)
            scheduler = ExponentialLR(opt, gamma=self.lr_scheduler_gamma)
            for epoch in range(self.min_epoch,self.max_epoch):
                lrs.append(scheduler.optimizer.param_groups[0]["lr"])
                opt.step()
                scheduler.step()
            self.optimizer.param_groups[0]['lr'] = lrs[self.start_epoch]
            self.scheduler = ExponentialLR(self.optimizer, gamma=self.lr_scheduler_gamma)
          else:
            cur_iters = self.start_epoch*len(self.train_dataloader)
            self.optimizer.param_groups[0]['lr'] = self.lr*((1-float(cur_iters)/self.max_iters)**(self.power))

          ## setting the best miou
          self.best_mIoU = wts['miou']



    def train(self):
        epochs, iters_per_epoch, max_iters = self.epochs, self.iters_per_epoch, self.max_iters
        #log_per_iters = self.cfg["train"]["log_iter"]
        #val_per_iters = self.cfg["train"]["val_epoch"] * self.iters_per_epoch

        start_time = time.time()
        self.logger.info('Start training, Total Epochs: {:d} = Iterations per epoch {:d}'.format(epochs, iters_per_epoch))

        self.model.train()
        for ep in range(self.start_epoch,self.end_epoch):
            self.current_epoch += 1
            lsit_pixAcc = []
            list_mIoU = []
            list_loss = []
            self.metric.reset()
            samplecnt = 0
            for i_iter, batch in enumerate(self.train_dataloader, 0):
                images, targets, bd_gts, _, _ = batch

                self.current_iteration += 1
                samplecnt += images.shape[0]

                images = images.to(self.device).float()
                targets = targets.to(self.device).long()
                bd_gts = bd_gts.to(self.device).float()

                # print(f'Image size: {images.shape}')
                # print(f'Image type: {images.dtype}')
                # print(f'target size: {targets.shape}')
                # print(f'target type: {targets.dtype}')

                outputs = self.model(images)
                del images
                # for i in outputs:
                #   print(i.shape)
                # print(f'target:{targets.shape}')
                # print(f'bd_gts:{bd_gts.shape}')
                loss = self.criterion(outputs,targets,bd_gts)
                #loss = losses.mean()
              

                # print(f'outputs size: {outputs[0].shape}')
                # print(f'outputs type: {outputs[0].dtype}')
                # print(f'targets size: {targets.shape}')
                # print(f'targets type: {targets.dtype}')

                self.metric.update(outputs[self.config.test.output_index], targets) # output from I branch
                pixAcc, mIoU = self.metric.get()
                lsit_pixAcc.append(pixAcc)
                list_mIoU.append(mIoU)
                list_loss.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                ## ETA
                eta_seconds = ((time.time() - start_time) / self.current_iteration) * (iters_per_epoch*epochs - self.current_iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                if self.current_iteration % self.log_per_iters == 0:
                  self.logger.info(
                      "Epochs: {:d}/{:d}/{:d} || Samples: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || mIoU: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                          self.start_epoch + self.current_epoch, self.end_epoch,self.max_epoch,
                          samplecnt, self.batch_train*iters_per_epoch,
                          self.optimizer.param_groups[0]['lr'],
                          loss.item(),
                          mIoU,
                          str(datetime.timedelta(seconds=int(time.time() - start_time))),
                          eta_string))

                # # ## adjusting lr
                if not self.lr_scheduler:
                  adjust_learning_rate(self.optimizer, 
                                      base_lr = self.lr, 
                                      max_iters = self.max_iters,
                                      cur_iters = ep*len(self.train_dataloader) + i_iter,
                                      power = self.power)

            #average_pixAcc = sum(lsit_pixAcc)/len(lsit_pixAcc)
            #average_mIoU = sum(list_mIoU)/len(list_mIoU)
            average_pixAcc, average_mIoU = self.metric.get()
            average_loss = sum(list_loss)/len(list_loss)
            self.logger.info('-------------------------------------------------------------------------------------------------')
            self.logger.info("Epochs: {:d}/{:d}, Average loss: {:.3f}, Average mIoU: {:.3f}, Average pixAcc: {:.3f}".format(
              self.current_epoch, self.epochs, average_loss, average_mIoU, average_pixAcc))

            
            self.test() ## Doing 1 iteration of testing
            self.logger.info('-------------------------------------------------------------------------------------------------')
            self.model.train() ## Setting the model back to train mode
            if self.lr_scheduler:
              self.scheduler.step()

        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        self.logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
            total_training_str, total_training_time / max_iters))

    def test(self):
        is_best = False
        self.metric.reset()
        self.model.eval()
        lsit_pixAcc = []
        list_mIoU = []
        list_loss = []
        for i, batch in enumerate(self.test_dataloader):
            images, targets,_, _, _ = batch
            images = images.to(self.device).float()
            targets = targets.to(self.device).long()

            with torch.no_grad():
                outputs = self.model(images)
                #loss = self.criterion(outputs, targets)
            self.metric.update(outputs[self.config.test.output_index], targets)
            #pixAcc, mIoU = self.metric.get()
            #lsit_pixAcc.append(pixAcc)
            #list_mIoU.append(mIoU)
            #list_loss.append(loss.item())

        #average_pixAcc = sum(lsit_pixAcc)/len(lsit_pixAcc)
        #average_mIoU = sum(list_mIoU)/len(list_mIoU)
        #average_loss = sum(list_loss)/len(list_loss)
        average_pixAcc, average_mIoU = self.metric.get()
        self.current_mIoU = average_mIoU
        self.logger.info("Testing: Average mIoU: {:.3f}, Average pixAcc: {:.3f}".format(average_mIoU, average_pixAcc))

        if self.current_mIoU > self.best_mIoU:
            is_best = True
            self.best_mIoU = self.current_mIoU
        if is_best:
            save_checkpoint(self.optimizer,self.model, self.logger,self.config, self.start_epoch+self.current_epoch, is_best, self.current_mIoU)

def save_checkpoint(optimizer,model, logger,cfg, epoch = 0, is_best=False, mIoU = 0.0):
    logger.info('Saving the checkpoint =>')
    directory = f"/content/drive/MyDrive/Colab Notebooks/1_Papers/2_RobustRealtimeSS/1_Pretraining_cityscape/1_PIDNet/experiments/{cfg.experiment.name}/checkpoints"
    filename = '{}_{}_{:.3f}.pth.tar'.format(get_eastern_time(),epoch,mIoU)
    filename = os.path.join(directory, filename)
    checkpoint = {
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'miou': mIoU,
    }
    torch.save(checkpoint,filename)

def get_eastern_time():
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z%z")

def adjust_learning_rate(optimizer, base_lr, max_iters, cur_iters, power=0.9):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
  
