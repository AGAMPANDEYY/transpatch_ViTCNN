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
import random
import torchvision.transforms.functional as TF
import torch.nn.functional as F

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
      self.criterion = PatchLoss(self.config, main_logger)

      ## optimizer
      # Initialize adversarial patch (random noise)
      # (A) Low-frequency via blur  (simple & solid default)
      self.patch = self.init_lowfreq_patch((3, self.patch_size, self.patch_size), k=31)
      
      # (B) Multi-octave Perlin-ish
      # self.patch = self.init_perlin_patch((3, self.patch_size, self.patch_size))
      
      # (C) Color-matched low-frequency (Cityscapes)
      # self.patch = self.init_color_matched_lowfreq((3, self.patch_size, self.patch_size), k=31)

      # ===== NEW: tanh-parameterized patch with low-frequency init =====
      #self.patch_param = self.init_lowfreq_tanh((3, self.patch_size, self.patch_size), cutoff=0.2)

      # (optional) choose another init by swapping the line above with:
      # self.patch_param = self.init_perlin_tanh((3, self.S, self.S))
      # self.patch_param = self.init_dataset_color_tanh((3, self.S, self.S), mean=(0.286,0.325,0.283))

      # Optimizer (Adam is typically more stable than per-step FGSM)
      #self.opt = torch.optim.Adam([self.patch_param], lr=self.lr)
      #self.scheduler = ExponentialLR(self.opt, gamma=self.lr_scheduler_gamma) if self.lr_scheduler else None

      # TV regularizer weight (can expose to config)
      #self.tv_weight = getattr(config.loss, 'tv_weight', 1e-4)

      
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


  def _normalize_01(self, x):
    # x: (C,H,W)
    mn = x.amin(dim=(-2,-1), keepdim=True)
    mx = x.amax(dim=(-2,-1), keepdim=True)
    return (x - mn) / (mx - mn + 1e-8)

  def init_lowfreq_patch(self, shape, k=31):
      """
      Very simple low-pass init: blur random noise with a big avg-pool kernel.
      Keeps more energy at low spatial freq; returns (C,H,W) in [0,1] with grad.
      """
      C,H,W = shape
      x = torch.rand(1, C, H, W, device=self.device)
      x = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k//2)  # low-pass
      x = x[0]
      x = self._normalize_01(x)
      x.requires_grad_(True)
      return x
  
  def init_perlin_patch(self, shape, octaves=(4,8,16,32), amps=(0.5,0.25,0.15,0.10)):
      """
      Procedural (Perlin-ish) multi-octave init using upsampled random grids.
      Returns (C,H,W) in [0,1] with grad.
      """
      C,H,W = shape
      base = 0.0
      for f,a in zip(octaves, amps):
          g = torch.rand(1, 1, f+1, f+1, device=self.device)  # coarse noise
          n = F.interpolate(g, size=(H,W), mode='bilinear', align_corners=True)
          base = base + a * n
      base = base[0,0]
      base = (base - base.min()) / (base.max() - base.min() + 1e-8)
      img = base.expand(C, H, W).contiguous()
      img.requires_grad_(True)
      return img
  
  def init_color_matched_lowfreq(self, shape, mean=(0.286,0.325,0.283), k=31, jitter=0.05):
      """
      Low-pass init biased to Cityscapes mean colors; helps photometric robustness.
      """
      C,H,W = shape
      mean = torch.tensor(mean, device=self.device)[:,None,None]
      noise = torch.rand(C, H, W, device=self.device) * (2*jitter) - jitter
      img = (mean + noise).clamp(0,1)[None]  # (1,C,H,W)
      img = F.avg_pool2d(img, kernel_size=k, stride=1, padding=k//2)[0]
      img.requires_grad_(True)
      return img

  # ---------------------
  # Patch parametrization & inits
  # ---------------------
  def get_patch(self):
      # maps R -> [0,1] smoothly and avoids exact 0/1 saturation
      return 0.5 * (torch.tanh(self.patch_param) + 1.0) * 0.999

  def init_lowfreq_tanh(self, shape, cutoff=0.2):
      """Low-frequency FFT init in [0,1], then inverse tanh to get patch_param."""
      C,H,W = shape
      device = self.device
      # random complex spectrum
      spec = torch.randn(C,H,W, dtype=torch.complex64, device=device)
      yy, xx = torch.meshgrid(
          torch.linspace(-1,1,H,device=device),
          torch.linspace(-1,1,W,device=device), indexing='ij')
      rad = (xx**2 + yy**2).sqrt()
      mask = (rad <= cutoff)
      spec = spec * mask  # keep low-freq only
      img = torch.fft.ifft2(spec).real
      # normalize to [0,1]
      img = (img - img.amin(dim=(-2,-1), keepdim=True))
      img = img / (img.amax(dim=(-2,-1), keepdim=True) - img.amin(dim=(-2,-1), keepdim=True) + 1e-8)
      # inverse tanh
      z = (img*2 - 1).clamp(-0.999, 0.999)
      param = torch.atanh(z).detach().to(device)
      param.requires_grad_(True)
      return param

  def init_perlin_tanh(self, shape):
      import torch.nn.functional as F
      C,H,W = shape; device = self.device
      def perlin_octave(freq, amp):
          grid = torch.rand(2, freq+1, freq+1, device=device)
          noise = F.interpolate(grid.unsqueeze(0), size=(H,W), mode='bilinear', align_corners=True)[0]
          # simple directional mix
          xs = torch.linspace(0,1,W,device=device)
          ys = torch.linspace(0,1,H,device=device)
          n = noise[0][None,:,:]*xs + noise[1][:,None]*ys
          return amp * (n - n.min()) / (n.max()-n.min()+1e-8)
      base = sum(perlin_octave(f,a) for f,a in [(4,0.5),(8,0.25),(16,0.15),(32,0.10)])
      base = base.clamp(0,1)
      img = torch.stack([base for _ in range(C)], dim=0)
      z = (img*2 - 1).clamp(-0.999, 0.999)
      param = torch.atanh(z).detach(); param.requires_grad_(True)
      return param.to(device)

  def init_dataset_color_tanh(self, shape, mean=(0.286,0.325,0.283)):
      C,H,W = shape; device = self.device
      mean = torch.tensor(mean, device=device)[:,None,None]
      lowf = (torch.rand(C,H,W, device=device)*0.1 - 0.05)
      img = (mean + lowf).clamp(0,1)
      z = (img*2 - 1).clamp(-0.999, 0.999)
      param = torch.atanh(z).detach(); param.requires_grad_(True)
      return param.to(device)

  # ---------------------
  # Light patch-space EOT (keeps patch size SxS)
  # ---------------------
  def eot_transform_patch(self, patch):
      # patch: (3,S,S) in [0,1]
      angle = random.uniform(-20, 20)
      scale = random.uniform(0.8, 1.2)
      shear = [random.uniform(-5,5), random.uniform(-5,5)]
      # affine keeps size; translate kept 0 because apply_patch chooses location
      patch_t = TF.affine(patch, angle=angle, translate=[0,0], scale=scale, shear=shear)
      # mild color jitter to simulate print/camera shifts
      patch_t = TF.adjust_brightness(patch_t, random.uniform(0.85, 1.15))
      patch_t = TF.adjust_contrast(patch_t,  random.uniform(0.85, 1.15))
      return patch_t.clamp(0,1)

  # ---------------------
  # Total variation for smoothness / physical robustness
  # ---------------------
  def tv_loss(self, x):
    if x.dim() == 4:  # (N,C,H,W)
        tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    elif x.dim() == 3:  # (C,H,W) -> your patch
        tv_h = (x[:, 1:, :] - x[:, :-1, :]).abs().mean()
        tv_w = (x[:, :, 1:] - x[:, :, :-1]).abs().mean()
    else:
        raise ValueError(f"tv_loss expects 3D or 4D tensor, got {x.dim()}D")
    return tv_h + tv_w

  def _segpgd_inner(self, image, label, T, alpha):
      """
      SegPGD inner loop that updates self.patch T times on *this batch*.
      image: (N,3,H,W), label: (N,H,W) long
      Returns: last (output, patched_label, loss_for_logging)
      Notes:
        - We disable patch-space EOT here (use raw self.patch) for stable gradients.
        - Patch placement stays whatever your Patch.apply_patch decides (e.g., center/random/fixed coords).
      """
      # We will use the anchor model only for the gradient (as in the paper);
      # you can still do ensemble in Stage-2 outside SegPGD mode if desired.
      loss_log = 0.0
      last_output = None
      valid_ignore = (label != self.config.train.ignore_label)
  
      for t in range(1, T+1):
          # paste current patch (no random EOT inside the inner loop)
          patched_image, patched_label = self.apply_patch(image, label, self.patch)
          patched_label = patched_label.to(self.device).long()
          valid = (patched_label != self.config.train.ignore_label)
  
          # forward
          output = self.model.predict(patched_image, patched_label.shape)  # (N,C,H,W)
          last_output = output
  
          # per-pixel CE
          ce = F.cross_entropy(output, patched_label, ignore_index=self.config.train.ignore_label, reduction='none')  # (N,H,W)
  
          # masks for correctly / wrongly predicted pixels
          with torch.no_grad():
              pred = output.argmax(dim=1)  # (N,H,W)
              correct = (pred == patched_label) & valid
              wrong   = (~correct) & valid
  
          # SegPGD schedule λ(t) = (t-1)/(2T)   (paper's simple linear schedule)
          lam = (t - 1.0) / (2.0 * T)
  
          # SegPGD loss: (1-λ)*CE_on_correct + λ*CE_on_wrong, normalized by #valid pixels
          num_valid = valid.sum().clamp_min(1).float()
          loss_t = ((1 - lam) * (ce[correct].sum()) + lam * (ce[wrong].sum())) / num_valid
  
          # gradient step on the patch (FGSM-style for speed; use sign; clamp to [0,1])
          self.model.model.zero_grad(set_to_none=True)
          if self.patch.grad is not None:
              self.patch.grad.zero_()
          loss_t.backward()
          with torch.no_grad():
              self.patch += alpha * self.patch.grad.sign()
              self.patch.clamp_(0, 1)
  
          loss_log += float(loss_t.item())
  
      # Return the last output and label so the outer loop can compute metrics/logging
      return last_output, patched_label, (loss_log / T)



  def train(self):
      # ----- epoch split (use config first; else fall back to an even split) -----
      start_epoch = int(self.start_epoch)
      end_epoch   = int(self.end_epoch)
      total_epochs = end_epoch - start_epoch
  
      E1_cfg = int(getattr(self.config.attack, "stage1_epochs", max(1, total_epochs // 2)))
      E2_cfg = int(getattr(self.config.attack, "stage2_epochs", total_epochs - E1_cfg))
      if E1_cfg + E2_cfg != total_epochs:
          # fallback to even split if the config doesn't add up
          E1_cfg = total_epochs // 2
          E2_cfg = total_epochs - E1_cfg
      switch_epoch = start_epoch + E1_cfg  # [start, switch) = Stage-1 ; [switch, end) = Stage-2
  
      iters_per_epoch = self.iters_per_epoch
      start_time = time.time()
  
      self.logger.info(
          f"Start training | Total={total_epochs} "
          f"(Stage-1: {start_epoch}–{switch_epoch-1} | Stage-2(JS): {switch_epoch}–{end_epoch-1}) | "
          f"iters/epoch={iters_per_epoch}"
      )
  
      use_segpgd = bool(getattr(self.config.attack, 'use_segpgd', False))
      seg_T      = int(getattr(self.config.attack, 'segpgd_steps', 3))
      seg_alpha  = float(getattr(self.config.attack, 'segpgd_alpha', self.epsilon))
  
      IoU = []
      for ep in range(start_epoch, end_epoch):
          self.current_epoch = ep
          self.metric.reset()
          if hasattr(self.criterion, "set_epoch"):
              self.criterion.set_epoch(ep)
  
          # epoch-wide stage choice (if not using SegPGD)
          use_stage1 = (ep < switch_epoch)
          stage_epoch_name = "Stage-1" if use_stage1 else "Stage-2(JS)"
          if not use_segpgd:
              self.logger.info(f"Epoch {ep}: using {stage_epoch_name}")
          else:
              self.logger.info(f"Epoch {ep}: using SegPGD[T={seg_T}, α={seg_alpha}]")
  
          total_loss = 0.0
          samplecnt = 0
  
          for i_iter, batch in enumerate(self.train_dataloader, 0):
              self.current_iteration += 1
  
              # ---- batch prep ----
              image, true_label, _, _, _ = batch
              image = image.to(self.device)
              true_label = true_label.to(self.device)
              samplecnt += image.shape[0]
  
              # make sure patch has grad for updates
              if not self.patch.requires_grad:
                  self.patch.requires_grad_(True)
  
              # ------------------- SegPGD mode -------------------
              if use_segpgd:
                  # inner loop handles: paste, forward, loss, and patch FGSM steps
                  output, patched_label, loss_val = self._segpgd_inner(
                      image, true_label, T=seg_T, alpha=seg_alpha
                  )
                  loss = torch.tensor(loss_val, device=self.device)  # for uniform logging
                  did_patch_update = True
  
              # ------------------- Stage-1 / Stage-2(JS) mode -------------------
              else:
                  # (optional) light EOT on the patch for physical robustness
                  patch_to_paste = self.eot_transform_patch(self.patch)
  
                  # paste & dtype
                  patched_image, patched_label = self.apply_patch(image, true_label, patch_to_paste)
                  patched_label = patched_label.to(self.device).long()
  
                  # forward anchor
                  output = self.model.predict(patched_image, patched_label.shape)
                  with torch.no_grad():
                      clean_output = self.model.predict(image, patched_label.shape)
  
                  # compute epoch-stage loss
                  if use_stage1:
                      loss = self.criterion.compute_loss_transegpgd_stage1(
                          output, patched_label, clean_output
                      )
                  else:
                      loss = self.criterion.compute_loss_transegpgd_stage2_js(
                          output, patched_label, clean_output
                      )
  
                  did_patch_update = False  # outer update will run below
  
              # accumulate for epoch log
              total_loss += float(loss.item())
  
              # metrics from anchor
              self.metric.update(output, patched_label)
              pixAcc, mIoU = self.metric.get()
  
              # ---- outer patch update (only when NOT using SegPGD) ----
              if not use_segpgd:
                  if hasattr(self.model, "model"):
                      self.model.model.zero_grad(set_to_none=True)
                  else:
                      self.model.zero_grad(set_to_none=True)
                  if self.patch.grad is not None:
                      self.patch.grad.zero_()
  
                  loss.backward()
                  with torch.no_grad():
                      # FGSM ascent
                      self.patch += self.epsilon * self.patch.grad.sign()
                      self.patch.clamp_(0, 1)
  
              # ---- logging ----
              if i_iter % self.log_per_iters == 0:
                  elapsed = int(time.time() - start_time)
                  eta = int((elapsed / max(self.current_iteration, 1)) *
                            (iters_per_epoch * total_epochs - self.current_iteration))
                  stage_tag = (
                      f"SegPGD[T={seg_T}]"
                      if use_segpgd else
                      stage_epoch_name
                  )
                  self.logger.info(
                      "Epoch: {:d}/{:d} || Stage:{} || Batch: {:d}/{:d} || "
                      "Samples: {:d}/{:d} || Step(ε): {:.6f} || Loss: {:.4f} || "
                      "mIoU: {:.4f} || Time: {} || ETA: {}".format(
                          self.current_epoch, end_epoch,
                          stage_tag, i_iter + 1, iters_per_epoch,
                          samplecnt, self.batch_train * iters_per_epoch,
                          self.epsilon, loss.item(), mIoU,
                          str(datetime.timedelta(seconds=elapsed)),
                          str(datetime.timedelta(seconds=eta))
                      )
                  )
  
          # ---- epoch summary ----
          avg_pixAcc, avg_mIoU = self.metric.get()
          avg_loss = total_loss / max(1, len(self.train_dataloader))
          self.logger.info('-' * 97)
          self.logger.info(
              "Epoch {:d}/{:d} | {} | Avg Loss: {:.4f} | Avg mIoU: {:.4f} | Avg pixAcc: {:.4f}".format(
                  self.current_epoch, end_epoch,
                  ("SegPGD" if use_segpgd else stage_epoch_name),
                  avg_loss, avg_mIoU, avg_pixAcc
              )
          )
          if hasattr(self.criterion, "log_epoch_summary"):
              self.criterion.log_epoch_summary()
          self.logger.info('-' * 97)
  
          IoU.append(self.metric.get(full=True))
  
      # return trained patch + IoU history
      return self.patch.detach(), np.array(IoU)
