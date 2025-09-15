import sys
original_sys_path = sys.path.copy()
sys.path.append("/kaggle/working/adversarial-patch-transferability/")

from dataset.cityscapes import Cityscapes
from pretrained_models.models import Models
from metrics.performance import SegmentationMetric
from metrics.loss import PatchLoss
from patch.create import Patch
from torch.optim.lr_scheduler import ExponentialLR

import time, torch, datetime, numpy as np, random, copy
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path = original_sys_path


class PatchTrainer():
  def __init__(self, config, main_logger):
      self.config = config
      self.start_epoch = config.train.start_epoch
      self.end_epoch   = config.train.end_epoch
      self.epochs      = self.end_epoch - self.start_epoch
      self.batch_train = config.train.batch_size
      self.batch_test  = config.test.batch_size
      self.device      = config.experiment.device
      self.logger      = main_logger
      self.lr          = config.optimizer.init_lr
      self.lr_scheduler= config.optimizer.exponentiallr
      self.lr_scheduler_gamma = config.optimizer.exponentiallr_gamma
      self.log_per_iters = config.train.log_per_iters
      self.patch_size   = config.patch.size
      self.apply_patch  = Patch(config).apply_patch
      self.epsilon      = config.optimizer.init_lr  # PGD step size

      # ----------------- DATA -----------------
      cityscape_train = Cityscapes(
          root=config.dataset.root, list_path=config.dataset.train,
          num_classes=config.dataset.num_classes, multi_scale=config.train.multi_scale,
          flip=config.train.flip, ignore_label=config.train.ignore_label,
          base_size=config.train.base_size,
          crop_size=(config.train.height, config.train.width),
          scale_factor=config.train.scale_factor)

      cityscape_val = Cityscapes(
          root=config.dataset.root, list_path=config.dataset.val,
          num_classes=config.dataset.num_classes, multi_scale=False, flip=False,
          ignore_label=config.train.ignore_label, base_size=config.test.base_size,
          crop_size=(config.test.height, config.test.width))

      self.train_dataloader = torch.utils.data.DataLoader(
          dataset=cityscape_train, batch_size=self.batch_train,
          shuffle=config.train.shuffle, num_workers=config.train.num_workers,
          pin_memory=config.train.pin_memory, drop_last=config.train.drop_last)

      self.test_dataloader = torch.utils.data.DataLoader(
          dataset=cityscape_val, batch_size=self.batch_test, shuffle=False,
          num_workers=config.test.num_workers, pin_memory=config.test.pin_memory,
          drop_last=config.test.drop_last)

      self.iters_per_epoch = len(self.train_dataloader)

      # ----------------- MODELS -----------------
      # Anchor model (metrics + stage switch)
      self.model = Models(self.config)
      self.model.get()
      self.model.model.eval()
      anchor_name = self.config.model.name

      # Ensemble of frozen surrogates (CNN + ViT) Surrogate Model
      self.ensemble = []
      default_names = ['pidnet_s', 'pidnet_m','pidnet_l', 'bisenet_v2', 'segformer']
      ens_names = getattr(self.config.attack, 'ensemble_names', default_names)

      for name in ens_names:
          if name == anchor_name:
              continue  # avoid duplicate; anchor is already included separately
          try:
              m = Models(self.config, model_name=name)  # <-- override here
              m.get()
              m.model.eval()
              for p in m.model.parameters():
                  p.requires_grad_(False)
              self.ensemble.append(m)
              self.logger.info(f"[Ensemble] added surrogate: {name}")
          except Exception as e:
              self.logger.info(f"[Ensemble] WARNING: could not load '{name}': {e}")

      self.ens_sample_size = int(getattr(self.config.attack, 'ens_k', 2))
      self.ens_tau         = float(getattr(self.config.attack, 'ens_tau', 0.1))  # smooth-max temp

      # ----------------- LOSS -----------------
      self.criterion = PatchLoss(self.config, main_logger)

      # ----------------- PATCH (PGD) -----------------
      # Better init (pick one). Default: low-frequency via blur.
      self.patch = self.init_lowfreq_patch((3, self.patch_size, self.patch_size), k=31)
      # # Alternative:
      # self.patch = self.init_perlin_patch((3, self.patch_size, self.patch_size))
      # self.patch = self.init_color_matched_lowfreq((3, self.patch_size, self.patch_size), k=31)

      # ----------------- METRICS/STATE -----------------
      self.metric = SegmentationMetric(config)
      self.current_epoch = 0
      self.current_iteration = 0

  # ---------- initializers ----------
  def _normalize_01(self, x):  # x: (C,H,W)
      mn = x.amin(dim=(-2,-1), keepdim=True)
      mx = x.amax(dim=(-2,-1), keepdim=True)
      return (x - mn) / (mx - mn + 1e-8)

  def init_lowfreq_patch(self, shape, k=31):
      C,H,W = shape
      x = torch.rand(1, C, H, W, device=self.device)
      x = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k//2)  # low-pass blur
      x = self._normalize_01(x[0])                                # (C,H,W) in [0,1]
      x.requires_grad_(True)
      return x

  def init_perlin_patch(self, shape, octaves=(4,8,16,32), amps=(0.5,0.25,0.15,0.10)):
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
      C,H,W = shape
      mean = torch.tensor(mean, device=self.device)[:,None,None]
      noise = torch.rand(C, H, W, device=self.device) * (2*jitter) - jitter
      img = (mean + noise).clamp(0,1)[None]  # (1,C,H,W)
      img = F.avg_pool2d(img, kernel_size=k, stride=1, padding=k//2)[0]
      img.requires_grad_(True)
      return img

  # ---------- patch-space EOT (optional) ----------
  def eot_transform_patch(self, patch):
      angle = random.uniform(-20, 20)
      scale = random.uniform(0.8, 1.2)
      shear = [random.uniform(-5,5), random.uniform(-5,5)]
      p = TF.affine(patch, angle=angle, translate=[0,0], scale=scale, shear=shear)
      p = TF.adjust_brightness(p, random.uniform(0.85, 1.15))
      p = TF.adjust_contrast(p,  random.uniform(0.85, 1.15))
      return p.clamp(0,1)

  # ---------- train ----------
  def train(self):
      epochs, iters_per_epoch = self.epochs, self.iters_per_epoch
      start_time = time.time()
      self.logger.info('Start training, Total Epochs: {:d} = Iterations per epoch {:d}'.format(epochs, iters_per_epoch))
      IoU = []

      for ep in range(self.start_epoch, self.end_epoch):
          self.current_epoch = ep
          self.metric.reset()
          if hasattr(self.criterion, "set_epoch"):
              self.criterion.set_epoch(ep)
              # >>> ADD (right after set_epoch(ep))
              E1  = int(getattr(self.config.attack, "stage1_epochs", 15))   # epochs for Stage-1
              tau = float(getattr(self.config.attack, "stage2_agree_thresh", 0.85))  # optional early switch

          total_loss = 0.0
          samplecnt = 0

          for i_iter, batch in enumerate(self.train_dataloader, 0):
              self.current_iteration += 1
              image, true_label, _, _, _ = batch
              image = image.to(self.device)
              true_label = true_label.to(self.device)
              samplecnt += image.shape[0]

              # patch (optionally EOT)
              patch_to_paste = self.eot_transform_patch(self.patch)

              # paste patch
              patched_image, patched_label = self.apply_patch(image, true_label, patch_to_paste)
              patched_label = patched_label.to(self.device).long()

              # anchor forward
              output = self.model.predict(patched_image, patched_label.shape)
              with torch.no_grad():
                  clean_output = self.model.predict(image, patched_label.shape)

              # decide stage using anchor
              with torch.no_grad():
                  pred_labels = output.argmax(dim=1)
                  correct_pixels = (pred_labels == patched_label) & (patched_label != self.config.train.ignore_label)
                  num_correct = int(correct_pixels.sum().item())

              # ---------- decide stage (epoch gate + optional agreement gate) ----------
              with torch.no_grad():
                  pred_adv   = output.argmax(dim=1)
                  pred_clean = clean_output.argmax(dim=1)
                  valid = (patched_label != self.config.train.ignore_label)
                  # fraction of valid pixels where adv == clean
                  agree_frac = ((pred_adv == pred_clean) & valid).float().mean().item()
              
              use_stage2 = (ep >= E1) #or (agree_frac <= tau)
              
              if use_stage2:
                  stage_name = f"S2-JS(agree={agree_frac:.3f})"
                  per_model_losses = []
              
                  # anchor contributes
                  loss_anchor = self.criterion.compute_loss_transegpgd_stage2_js(
                      output, patched_label, clean_output
                  )
                  per_model_losses.append(loss_anchor)
                  sampled_names = ["anchor"]
              
                  # sample K surrogates
                  if len(self.ensemble) > 0 and self.ens_sample_size > 0:
                      models_k = random.sample(self.ensemble, k=min(self.ens_sample_size, len(self.ensemble)))
                  else:
                      models_k = []
              
                  for m in models_k:
                      out_m = m.predict(patched_image, patched_label.shape)
                      with torch.no_grad():
                          clean_m = m.predict(image, patched_label.shape)
                      per_model_losses.append(
                          self.criterion.compute_loss_transegpgd_stage2_js(out_m, patched_label, clean_m)
                      )
                      sampled_names.append(getattr(m.config.model, "name", "unk"))
              
                  # aggregate across models: smooth max if tau>0 else average
                  if len(per_model_losses) == 1:
                      loss = per_model_losses[0]
                  elif self.ens_tau > 0:
                      Ls = torch.stack(per_model_losses)
                      loss = self.ens_tau * torch.logsumexp(Ls / self.ens_tau, dim=0) \
                             - self.ens_tau * torch.log(torch.tensor(len(per_model_losses), device=Ls.device, dtype=Ls.dtype))
                  else:
                      loss = torch.stack(per_model_losses).mean()
              
              else:
                  stage_name = f"S1(agree={agree_frac:.3f})"
                  sampled_names = ["anchor"]
                  loss = self.criterion.compute_loss_transegpgd_stage1(output, patched_label, clean_output)

              total_loss += float(loss.item())

              # metrics on anchor
              self.metric.update(output, patched_label)
              pixAcc, mIoU = self.metric.get()

              # --- PGD update on patch ---
              self.model.model.zero_grad()
              if self.patch.grad is not None:
                  self.patch.grad.zero_()
              loss.backward()
              with torch.no_grad():
                  self.patch += self.epsilon * self.patch.grad.sign()
                  self.patch.clamp_(0, 1)

              # LOG
              if i_iter % self.log_per_iters == 0:
                  elapsed = int(time.time() - start_time)
                  eta = int((elapsed / max(self.current_iteration,1)) * (iters_per_epoch*epochs - self.current_iteration))
                  self.logger.info(
                      "Epochs: {:d}/{:d} || Stage:{} || Models:{} || Samples: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || mIoU: {:.4f} || Time: {} || ETA: {}".format(
                          self.current_epoch, self.end_epoch,
                          stage_name, ",".join(sampled_names),
                          samplecnt, self.batch_train*iters_per_epoch,
                          self.epsilon, loss.item(), mIoU,
                          str(datetime.timedelta(seconds=elapsed)),
                          str(datetime.timedelta(seconds=eta))
                      )
                  )

          avg_pixAcc, avg_mIoU = self.metric.get()
          avg_loss = total_loss / max(len(self.train_dataloader), 1)
          self.logger.info('-------------------------------------------------------------------------------------------------')
          self.logger.info("Epochs: {:d}/{:d}, Average loss: {:.3f}, Average mIoU: {:.3f}, Average pixAcc: {:.3f}".format(
              self.current_epoch, self.epochs, avg_loss, avg_mIoU, avg_pixAcc))
          if hasattr(self.criterion, "log_epoch_summary"):
              self.criterion.log_epoch_summary()
          self.logger.info('-------------------------------------------------------------------------------------------------')

          IoU.append(self.metric.get(full=True))

      return self.patch.detach(), np.array(IoU)
