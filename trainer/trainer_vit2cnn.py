import sys, time, datetime, random, copy
from typing import Tuple
original_sys_path = sys.path.copy()
sys.path.append("/kaggle/working/adversarial-patch-transferability/")

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np

from dataset.cityscapes import Cityscapes
from metrics.performance import SegmentationMetric
from metrics.loss import PatchLoss
from patch.create import Patch
from torch.optim.lr_scheduler import ExponentialLR

# Surrogate CNN (optional: for grad alignment)
from pretrained_models.PIDNet.model import get_pred_model  # assumes available in your repo

# Hugging Face SegFormer (ViT-backbone)
from transformers import AutoConfig, SegformerForSemanticSegmentation
from torch.cuda.amp import autocast

# Restore original sys.path
sys.path = original_sys_path


class PatchTrainer:
    """
    ViT-target patch training with:
      - Attention Hijack (ViT)
      - Boundary Disruption (clean vs patched edge divergence)
      - Frequency Shaping (encourage mid/high frequency energy)
      - Optional Surrogate-CNN Gradient Alignment (2nd-order)
      - EOT, TV, PGD/Adam

    Loss (minimize):
      total = -L_attack
              + tv_w * TV
              + attn_w * L_attnHijack
              + boundary_w * L_boundary      (returns negative to maximize edge divergence)
              + freq_w * L_freq              (returns negative to maximize ring energy)
              + grad_align_w * L_grad_align  (cosine-based; needs surrogate)
    """

    def __init__(self, config, main_logger):
        self.cfg = config
        self.log = main_logger
        self.device = config.experiment.device

        # ---------------- Dataloaders ----------------
        cityscape_train = Cityscapes(
            root=config.dataset.root,
            list_path=config.dataset.train,
            num_classes=config.dataset.num_classes,
            multi_scale=config.train.multi_scale,
            flip=config.train.flip,
            ignore_label=config.train.ignore_label,
            base_size=config.train.base_size,
            crop_size=(config.train.height, config.train.width),
            scale_factor=config.train.scale_factor,
        )
        cityscape_val = Cityscapes(
            root=config.dataset.root,
            list_path=config.dataset.val,
            num_classes=config.dataset.num_classes,
            multi_scale=False,
            flip=False,
            ignore_label=config.train.ignore_label,
            base_size=config.test.base_size,
            crop_size=(config.test.height, config.test.width),
        )
        self.train_dl = torch.utils.data.DataLoader(
            cityscape_train, batch_size=config.train.batch_size,
            shuffle=config.train.shuffle, num_workers=config.train.num_workers,
            pin_memory=config.train.pin_memory, drop_last=config.train.drop_last)
        self.val_dl = torch.utils.data.DataLoader(
            cityscape_val, batch_size=config.test.batch_size,
            shuffle=False, num_workers=config.test.num_workers,
            pin_memory=config.test.pin_memory, drop_last=config.test.drop_last)

        self.iters_per_epoch = len(self.train_dl)
        self.start_epoch = config.train.start_epoch
        self.end_epoch   = config.train.end_epoch
        self.total_epochs = self.end_epoch - self.start_epoch
        assert self.total_epochs > 0

        # ---------------- SegFormer (ViT target) ----------------
        hf_name = getattr(config.model, "hf_name",
                          "nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
        hf_cfg = AutoConfig.from_pretrained(
            hf_name, num_labels=config.dataset.num_classes,
            output_attentions=True, output_hidden_states=False
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(hf_name, config=hf_cfg)
        self.model.to(self.device).eval()
        for p in self.model.parameters(): p.requires_grad_(False)

        # ---------------- Optional Surrogate CNN ----------------
        self.use_surrogate = bool(getattr(getattr(config, "surrogate", object()), "enable", False))
        self.surrogate = None
        if self.use_surrogate:
            s_name = getattr(config.surrogate, "name", "pidnet_s")
            try:
                self.surrogate = get_pred_model(s_name, num_classes=config.dataset.num_classes)
                self.surrogate.to(self.device).eval()
                for p in self.surrogate.parameters(): p.requires_grad_(False)
                self.log.info(f"[Surrogate] Loaded CNN: {s_name}")
            except Exception as e:
                self.log.info(f"[Surrogate] Failed to load '{s_name}': {e}. Disabling grad alignment.")
                self.use_surrogate = False

        # ---------------- Losses / patch / optimizer ----------------
        self.criterion = PatchLoss(config, main_logger)

        self.S = config.patch.size
        self.lr = config.optimizer.init_lr
        self.use_pgd = bool(getattr(config.optimizer, "use_pgd", False))
        self.pgd_steps = getattr(config.optimizer, "pgd_steps", 7)
        self.pgd_alpha = getattr(config.optimizer, "pgd_alpha", 2.0/255.0)

        # tanh-param patch
        self.patch_param = self.init_lowfreq_tanh((3, self.S, self.S), cutoff=0.2).to(self.device)
        if not self.use_pgd:
            self.opt = torch.optim.Adam([self.patch_param], lr=self.lr)
        else:
            self.opt = None

        self.lr_scheduler = config.optimizer.exponentiallr
        self.gamma = config.optimizer.exponentiallr_gamma
        self.scheduler = ExponentialLR(self.opt, gamma=self.gamma) if (self.lr_scheduler and self.opt) else None

        # weights
        self.tv_weight       = getattr(getattr(config, "loss", object()), "tv_weight", 1e-4)
        self.attn_w          = getattr(getattr(config, "loss", object()), "attn_hijack_w", 0.10)
        self.boundary_w      = getattr(getattr(config, "loss", object()), "boundary_w", 0.20)
        self.freq_w          = getattr(getattr(config, "loss", object()), "freq_w", 0.05)
        self.grad_align_w    = getattr(getattr(config, "loss", object()), "grad_align_w", 0.10)
        # frequency ring
        self.r_low           = getattr(getattr(config, "freq", object()), "r_low", 0.12)
        self.r_high          = getattr(getattr(config, "freq", object()), "r_high", 0.45)

        self.metric = SegmentationMetric(config)
        self.log_every = config.train.log_per_iters

        self.apply_patch = Patch(config).apply_patch
        self.ignore_index = config.train.ignore_label
        self.num_classes  = config.dataset.num_classes
        
        # --- Sidewalk placement options (Cityscapes: sidewalk=1) ---
        self.sidewalk_id           = getattr(config.dataset, "sidewalk_id", 5)
        self.place_on_sidewalk     = getattr(getattr(config, "patch", object()), "place_on_sidewalk", True)
        self.sidewalk_entropy_bias = getattr(getattr(config, "patch", object()), "sidewalk_entropy_bias", True)
        self.sidewalk_topk_frac    = float(getattr(getattr(config, "patch", object()), "sidewalk_topk_frac", 0.20))  # top 20%
        self.sidewalk_dilate       = int(getattr(getattr(config, "patch", object()), "sidewalk_dilate", 5))          # pixels
        self.mask_patch_labels     = bool(getattr(getattr(config, "patch", object()), "mask_patch_labels", False))   # set ignore_index under patch



    def _softmax_entropy(self, logits, dim=1, eps=1e-8):
        p = torch.softmax(logits, dim=dim).clamp_min(eps)
        return -(p * p.log()).sum(dim=dim)  # (B,H,W)
    
    def _dilate_mask(self, mask_2d, k=5):
        if k <= 1:
            return mask_2d
        m = mask_2d.float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        m = F.max_pool2d(m, kernel_size=k, stride=1, padding=k//2)
        return (m[0,0] > 0.5)
    
    def _choose_patch_topleft_from_mask(self, mask_2d, S, entropy_2d=None, topk_frac=0.2):
        H, W = mask_2d.shape
        valid = mask_2d.clone()
        if S > 1:
            valid[:S//2, :] = False
            valid[H-(S-S//2):, :] = False
            valid[:, :S//2] = False
            valid[:, W-(S-S//2):] = False
        idx = valid.nonzero(as_tuple=False)  # (K,2): [y,x]
        if idx.numel() == 0:
            return None
        if entropy_2d is not None:
            ent = entropy_2d[idx[:,0], idx[:,1]]
            k = max(1, int(topk_frac * idx.size(0)))
            _, topk_inds = torch.topk(ent, k, largest=True)
            idx = idx[topk_inds]
        yx = idx[torch.randint(0, idx.size(0), (1,), device=idx.device).item()]
        yc, xc = int(yx[0]), int(yx[1])
        y0 = max(0, min(yc - S//2, H - S))
        x0 = max(0, min(xc - S//2, W - S))
        return y0, x0
    
    def _paste_patch(self, img_bchw, patch_3chw, y0, x0):
        out = img_bchw.clone()
        S = patch_3chw.shape[-1]
        out[:, :, y0:y0+S, x0:x0+S] = patch_3chw[None]
        return out

    # ---------------- Patch parametrization ----------------
    def get_patch(self):
        return 0.5 * (torch.tanh(self.patch_param) + 1.0) * 0.999

    def init_lowfreq_tanh(self, shape, cutoff=0.2):
        C,H,W = shape
        spec = torch.randn(C, H, W, dtype=torch.complex64, device=self.device)
        yy, xx = torch.meshgrid(torch.linspace(-1,1,H,device=self.device),
                                torch.linspace(-1,1,W,device=self.device), indexing="ij")
        rad = (xx**2 + yy**2).sqrt()
        spec = spec * (rad <= cutoff)
        img = torch.fft.ifft2(spec).real
        img = (img - img.amin(dim=(-2,-1), keepdim=True))
        img = img / (img.amax(dim=(-2,-1), keepdim=True) - img.amin(dim=(-2,-1), keepdim=True) + 1e-8)
        z = (img*2 - 1).clamp(-0.999, 0.999)
        param = torch.atanh(z).detach(); param.requires_grad_(True)
        return param

    # ---------------- EOT on patch ----------------
    def eot_patch(self, patch_3chw):
        angle = random.uniform(-20, 20)
        scale = random.uniform(0.85, 1.15)
        shear = [random.uniform(-5,5), random.uniform(-5,5)]
        out = TF.affine(patch_3chw, angle=angle, translate=[0,0], scale=scale, shear=shear)
        out = TF.adjust_brightness(out, random.uniform(0.85, 1.15))
        out = TF.adjust_contrast(out,  random.uniform(0.85, 1.15))
        return out.clamp(0,1)

    # ---------------- Regularizers ----------------
    def tv_loss(self, x):
        if x.dim()==3:
            return (x[:,1:,:]-x[:,:-1,:]).abs().mean() + (x[:,:,1:]-x[:,:,:-1]).abs().mean()
        elif x.dim()==4:
            return (x[:,:,1:,:]-x[:,:,:-1,:]).abs().mean() + (x[:,:,:,1:]-x[:,:,:,:-1]).abs().mean()
        else:
            raise ValueError("tv_loss expects 3D/4D")

    @torch.no_grad()
    def _estimate_patch_mask(self, clean_img, patched_img, thresh=1e-3):
        diff = (patched_img - clean_img).abs().sum(dim=1, keepdim=True)  # (B,1,H,W)
        return (diff > thresh).float()

    def saliency_hijack_loss(self, logits_adv, patched_image, patch_mask) -> torch.Tensor:
            """
            Fallback when attentions are unusable:
            Maximize gradient magnitude inside the patch vs outside.
            """
            # we’ll use entropy of the softmax as a generic objective to derive saliency
            probs = torch.softmax(logits_adv, dim=1)
            entropy = -(probs * (probs.clamp_min(1e-8)).log()).sum(dim=1).mean()  # scalar
        
            # need gradients of entropy w.r.t. the input image
            # ensure graph exists through patched_image
            if not patched_image.requires_grad:
                patched_image.requires_grad_(True)
        
            grad = torch.autograd.grad(
                entropy, patched_image, create_graph=True, retain_graph=True, allow_unused=False
            )[0]  # (B,3,H,W)
        
            gmag = grad.abs().mean(dim=1, keepdim=True)  # (B,1,H,W)
        
            # maximize inside-patch saliency, minimize outside-patch
            inside = (gmag * patch_mask).mean()
            outside = (gmag * (1.0 - patch_mask)).mean()
            # Negative of contrast → minimizing total increases (inside - outside)
            return -(inside - outside)

    ##-------------------------- ViT CNN alignment ------------------- #
    def logit_agreement_loss(self, vit_logits, cnn_logits):
        """
        First-order surrogate shaping: maximize cosine similarity between ViT and CNN logits.
        Shapes: (B, C, H, W) for both; both already resized to target_hw.
        """
        v = vit_logits.flatten(1)
        c = cnn_logits.flatten(1)
        v = F.normalize(v, dim=1)
        c = F.normalize(c, dim=1)
        # negative cosine => minimizing pushes them to be similar
        return - (v * c).sum(dim=1).mean()
    
    def kl_align(self, p_logits, q_logits, T=1.0):
        """
        Alternative first-order option: KL(p || q).
        """
        p = F.log_softmax(p_logits / T, dim=1)
        q = F.softmax(q_logits / T, dim=1)
        return F.kl_div(p, q, reduction='batchmean') * (T * T)
    
    def patch_entropy_loss(self, logits, patch_mask):
        """
        Optional attention-free hijack: maximize entropy only under the patch (first-order).
        """
        probs = torch.softmax(logits, dim=1)
        ent = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)  # (B,1,H,W)
        return - (ent * patch_mask).mean()  # negative => maximize entropy under patch


    def attn_hijack_loss(self, attentions, patch_mask, H: int, W: int) -> torch.Tensor:
            """
            Robust attention hijack: try to maximize attention mass to patch tokens.
            If no compatible attention blocks exist, returns a scalar 0 tensor.
            (We'll add a saliency fallback at the call site.)
            """
            device = self.device
            if (attentions is None) or (len(attentions) == 0):
                return torch.zeros((), device=device)
        
            B = patch_mask.size(0)
            strides = [4, 8, 16, 32]  # common MiT stages
            hijack_sum, used = 0.0, 0
        
            for att in attentions:
                if att is None or not torch.is_tensor(att) or att.ndim != 4:
                    continue
                B_a, n_heads, Nq, Nk = att.shape
                # must be batch-aligned and square self-attention
                if (B_a != B) or (Nq != Nk) or (Nq <= 0):
                    continue
        
                # find token grid that matches this N
                stride = None
                for s in strides:
                    if (H // s) * (W // s) == Nq:
                        stride = s
                        break
                if stride is None:
                    # fallback: try square root grid
                    root = int(round(Nq ** 0.5))
                    if root * root != Nq:
                        continue
                    h_s = w_s = root
                else:
                    h_s, w_s = H // stride, W // stride
        
                pmask = F.interpolate(patch_mask, size=(h_s, w_s), mode="nearest").view(B, -1)  # (B, N)
                att_mean = att.mean(dim=1)  # (B, N, N)
        
                batch_loss, valid = 0.0, 0
                for b in range(B):
                    cols = pmask[b] > 0.5
                    if cols.sum() == 0:
                        continue
                    mass_to_patch = att_mean[b][:, cols].mean()
                    batch_loss += (-mass_to_patch)  # negative → maximize attention to patch
                    valid += 1
        
                if valid > 0:
                    hijack_sum += (batch_loss / valid)
                    used += 1
        
            if used == 0:
                return torch.zeros((), device=device)
            return hijack_sum / used


    def boundary_disruption_loss(self, clean_logits, adv_logits) -> torch.Tensor:
        # Argmax → edges via Sobel → maximize L1 difference (return negative to maximize)
        with torch.no_grad():
            clean = clean_logits.argmax(dim=1, keepdim=True).float()  # (B,1,H,W)
        adv = adv_logits.argmax(dim=1, keepdim=True).float()
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=adv.device, dtype=adv.dtype).view(1,1,3,3)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=adv.device, dtype=adv.dtype).view(1,1,3,3)
        e_c = F.conv2d(clean, kx, padding=1).abs() + F.conv2d(clean, ky, padding=1).abs()
        e_a = F.conv2d(adv,   kx, padding=1).abs() + F.conv2d(adv,   ky, padding=1).abs()
        return - F.l1_loss(e_a, e_c)  # negative → maximizing difference

    def freq_shaping_loss(self, patch_3chw) -> torch.Tensor:
        """
        Encourage energy in a radial frequency ring [r_low, r_high] (normalized radius 0..1).
        Returns negative mean magnitude in ring (so minimizing total increases energy).
        """
        C, H, W = patch_3chw.shape
        x = patch_3chw - patch_3chw.mean(dim=(-2,-1), keepdim=True)  # zero-mean per channel
        spec = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=(-2,-1))
        mag = spec.abs().mean(dim=0)  # (H,W) average over channels
        yy, xx = torch.meshgrid(torch.linspace(-1,1,H,device=x.device),
                                torch.linspace(-1,1,W,device=x.device), indexing="ij")
        rad = (xx**2 + yy**2).sqrt()
        ring = (rad>=self.r_low) & (rad<=self.r_high)
        if ring.sum()==0:
            return torch.zeros((), device=x.device)
        band_energy = mag[ring].mean()
        return - band_energy

    # ---------------- Forwards ----------------
    def segformer_forward(self, img_4bhwc, out_size=None):
            out = self.model(img_4bhwc, output_attentions=True)
            logits = out.logits  # (B,C,h,w)
            if out_size is not None and logits.shape[-2:] != out_size:
                logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
            return logits, out.attentions

    def _pick_highres_4d(self, tensors):
        """Pick the 4D tensor with the largest H*W (highest resolution)."""
        best = None; best_area = -1
        for t in tensors:
            if torch.is_tensor(t) and t.ndim == 4:
                h, w = t.shape[-2], t.shape[-1]
                area = int(h) * int(w)
                if area > best_area:
                    best = t; best_area = area
        return best
    
    def _extract_logits(self, out):
        """
        Normalize various model outputs to a single (B,C,H,W) tensor.
        Supports: Tensor, list/tuple of tensors, dict with tensor values, or objects with .logits.
        Returns None if nothing usable is found.
        """
        # direct tensor
        if torch.is_tensor(out):
            return out
    
        # HF-style: object with .logits
        if hasattr(out, "logits") and torch.is_tensor(out.logits):
            return out.logits
    
        # dict: try common keys, else any tensor value
        if isinstance(out, dict):
            for k in ("logits", "out", "main_out", "pred", "aux", "output"):
                v = out.get(k, None)
                if torch.is_tensor(v):
                    return v
            # fallback: any 4D tensor value
            return self._pick_highres_4d([v for v in out.values() if torch.is_tensor(v)])
    
        # list/tuple: multi-scale outputs -> pick highest-res 4D
        if isinstance(out, (list, tuple)):
            return self._pick_highres_4d(list(out))
    
        # unknown container
        return None
    
    def surrogate_forward_logits(self, img_4bhwc, target_size):
        """
        Forward surrogate and return a (B,C,H,W) tensor resized to target_size.
        Handles multi-scale outputs and dict/list returns.
        """
        if self.surrogate is None:
            return None
    
        out = self.surrogate(img_4bhwc)
    
        logits = self._extract_logits(out)
        if logits is None:
            # optional: uncomment to debug different PIDNet heads
            # self.log.info(f"[Surrogate] Could not extract logits from type {type(out)}")
            return None
    
        # Some models may return (B,H,W) with implicit C=1; expand
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)
        elif logits.ndim != 4:
            return None
    
        if logits.shape[-2:] != target_size:
            logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
        return logits

    # ---------------- Train ----------------
    def train(self):
        #start_epoch, end_epoch, total_epochs = self.start_epoch, self.end_epoch, self.total_epochs
        #assert total_epochs == 30, f"This schedule expects 30 epochs; got {total_epochs}."
        #switch_epoch = start_epoch + (total_epochs // 2)
        desired_total = int(getattr(self.cfg.train, "total_epochs", 20))  # 20
        self.end_epoch = self.start_epoch + desired_total
        start_epoch, end_epoch = self.start_epoch, self.end_epoch
        total_epochs = end_epoch - start_epoch
        switch_epoch = start_epoch + (total_epochs // 2)  # 10/10 split if total=20

        start_time = time.time()
        self.log.info(
            f"Start training | Total Epochs: {total_epochs} "
            f"(Stage-1: {start_epoch}–{switch_epoch-1}, Stage-2(JS): {switch_epoch}–{end_epoch-1}) | "
            f"Iterations/epoch: {self.iters_per_epoch}"
        )

        max_batches = int(getattr(self.cfg.train, "max_batches_per_epoch", 300))  # 300 << 2975
        max_epochs  = int(getattr(self.cfg.train, "max_epochs", 10))              # 10 << 30
        self.end_epoch = min(self.end_epoch, self.start_epoch + max_epochs)
        self.log.info(f"[Limiter] max_batches_per_epoch={max_batches}, max_epochs={max_epochs}")
        surrogate_every = int(getattr(self.cfg.train, "surrogate_every", 4))  # every 4th step

        IoU_over_epochs = []
        H, W = self.cfg.train.height, self.cfg.train.width

        for ep in range(start_epoch, end_epoch):
            self.metric.reset()
            use_stage1 = (ep < switch_epoch)
            stage = "Stage-1" if use_stage1 else "Stage-2(JS)"
            self.log.info(f"Epoch {ep}: using {stage}")

            cum_attack_loss = 0.0
            for it, batch in enumerate(self.train_dl, 0):
                if max_batches and it >= max_batches:
                        break
                image, true_label, _, _, _ = batch
                image = image.to(self.device)
                true_label = true_label.to(self.device).long()

                #base_patch = self.get_patch()
                #patch = self.eot_patch(base_patch)

                #patched_image, patched_label = self.apply_patch(image, true_label, patch)
                #patched_label = patched_label.long()
                #SIDEWALK P[ATCH PLACEMENT 

                base_patch = self.get_patch()
                patch = self.eot_patch(base_patch)
                S = patch.shape[-1]
                target_hw = true_label.shape[-2:]
                
                # --- get clean logits ON THE CLEAN IMAGE (you already have downscale logic below; reuse it) ---
                vit_ds = float(getattr(self.cfg.train, "vit_downscale", 0.75))
                if vit_ds < 1.0:
                    ds_hw = (int(image.shape[-2] * vit_ds), int(image.shape[-1] * vit_ds))
                    image_ds = F.interpolate(image, size=ds_hw, mode="bilinear", align_corners=False)
                else:
                    image_ds = image
                
                with torch.no_grad(), autocast(dtype=torch.float16):
                    clean_logits, _ = self.segformer_forward(image_ds, out_size=target_hw)  # (B,C,H,W)
                
                # --- build sidewalk mask from clean prediction ---
                clean_pred = clean_logits.argmax(dim=1)                # (B,H,W)
                sidewalk_mask = (clean_pred == self.sidewalk_id)       # (B,H,W)
                
                # optional entropy bias on sidewalk pixels
                ent = self._softmax_entropy(clean_logits) if self.sidewalk_entropy_bias else None  # (B,H,W)
                
                # optional dilation (per image)
                if self.sidewalk_dilate > 1:
                    sidewalk_mask = torch.stack([self._dilate_mask(m, self.sidewalk_dilate) for m in sidewalk_mask], dim=0)
                
                # --- choose per-image location & paste the same patch ---
                patched_imgs = []
                patched_label = true_label.clone().long()
                for b in range(image.size(0)):
                    yx = self._choose_patch_topleft_from_mask(
                        sidewalk_mask[b], S,
                        entropy_2d=(ent[b] if ent is not None else None),
                        topk_frac=self.sidewalk_topk_frac
                    )
                    if yx is None:  # fallback random
                        Htot, Wtot = target_hw
                        y0 = random.randint(0, max(0, Htot - S))
                        x0 = random.randint(0, max(0, Wtot - S))
                    else:
                        y0, x0 = yx
                
                    # paste the EOT patch
                    patched_imgs.append(self._paste_patch(image[b:b+1], patch, y0, x0))
                
                    # (optional) mask labels under the patch to ignore supervision there
                    if self.mask_patch_labels:
                        patched_label[b, y0:y0+S, x0:x0+S] = self.ignore_index
                
                patched_image = torch.cat(patched_imgs, dim=0)


               # --- Downscale INTO SegFormer, keep losses/metrics at label size ---
                target_hw = patched_label.shape[-2:]  # (H,W) of labels
                vit_ds = float(getattr(self.cfg.train, "vit_downscale", 0.75))  # 0.75 is a good default
                if vit_ds < 1.0:
                    ds_hw = (int(image.shape[-2] * vit_ds), int(image.shape[-1] * vit_ds))
                    image_ds         = F.interpolate(image,         size=ds_hw, mode="bilinear", align_corners=False)
                    patched_image_ds = F.interpolate(patched_image, size=ds_hw, mode="bilinear", align_corners=False)
                else:
                    image_ds, patched_image_ds = image, patched_image

                # ViT forward (resized to label size and Mixed precision)
                with autocast(dtype=torch.float16):
                    logits_adv, atts_adv = self.segformer_forward(patched_image_ds, out_size=target_hw)
                with torch.no_grad(), autocast(dtype=torch.float16):
                    logits_clean, _ = self.segformer_forward(image_ds, out_size=target_hw)


                # Attack loss (your two-stage)
                if use_stage1:
                    attack_loss = self.criterion.compute_loss_transegpgd_stage1(
                        logits_adv, patched_label, logits_clean
                    )
                else:
                    attack_loss = self.criterion.compute_loss_transegpgd_stage2_js(
                        logits_adv, patched_label, logits_clean
                    )

                # Regularizers
                with torch.no_grad():
                    patch_mask = self._estimate_patch_mask(image, patched_image)
                    
                ah_loss = self.attn_hijack_loss(atts_adv, patch_mask, H, W)               # attention hijack
                
                if ah_loss.numel() == 0 or (ah_loss.detach().abs() < 1e-12):
                    # fallback: saliency hijack (model-agnostic)
                    # NOTE: saliency_hijack needs gradients through `patched_image`
                    patched_image.requires_grad_(True)
                    ah_loss = self.saliency_hijack_loss(logits_adv, patched_image, patch_mask)

                tv = self.tv_loss(base_patch)                                            # TV
                b_loss = self.boundary_disruption_loss(logits_clean, logits_adv)         # boundary/region
                f_loss = self.freq_shaping_loss(base_patch)                              # frequency ring

                # Optional: Surrogate-CNN gradient alignment (1st-order grads)
    
                ga_loss = torch.zeros((), device=self.device)
                use_sur_now = (self.use_surrogate and self.grad_align_w > 0.0 and
                               (surrogate_every <= 1 or (it % surrogate_every == 0)))
                if use_sur_now:
                    sur_logits = self.surrogate_forward_logits(patched_image, target_size=target_hw)
                    if sur_logits is not None:
                        ga_loss = self.logit_agreement_loss(logits_adv.detach(), sur_logits)
                        
                        # Option 2: KL alignment (comment out option 1 to use this)
                        # ga_loss = self.kl_align(logits_adv, sur_logits, T=1.0)

                # Total (minimize)
                total = (-attack_loss) \
                        + self.tv_weight * tv \
                        + self.attn_w * ah_loss \
                        + self.boundary_w * b_loss \
                        + self.freq_w * f_loss \
                        + self.grad_align_w * ga_loss

                # ---- Optimize patch ----
                if not self.use_pgd:
                    self.model.zero_grad(set_to_none=True)
                    if self.surrogate is not None: self.surrogate.zero_grad(set_to_none=True)
                    if self.patch_param.grad is not None: self.patch_param.grad.zero_()
                    total.backward()
                    if self.opt is not None: self.opt.step()
                    if self.scheduler: self.scheduler.step()
                else:
                    K, alpha = self.pgd_steps, self.pgd_alpha
                    for _ in range(K):
                        base_patch = self.get_patch()
                        patch = self.eot_patch(base_patch)
                        patched_image, patched_label = self.apply_patch(image, true_label, patch)
                        patched_label = patched_label.long()

                        target_hw = patched_label.shape[-2:]

                        # --- Downscale INTO SegFormer here too ---
                        vit_ds = float(getattr(self.cfg.train, "vit_downscale", 0.75))
                        if vit_ds < 1.0:
                            ds_hw = (int(image.shape[-2] * vit_ds), int(image.shape[-1] * vit_ds))
                            image_ds         = F.interpolate(image,         size=ds_hw, mode="bilinear", align_corners=False)
                            patched_image_ds = F.interpolate(patched_image, size=ds_hw, mode="bilinear", align_corners=False)
                        else:
                            image_ds, patched_image_ds = image, patched_image

                        logits_adv, atts_adv = self.segformer_forward(patched_image, out_size=target_hw)
                        with torch.no_grad():
                            logits_clean, _    = self.segformer_forward(image,        out_size=target_hw)


                        if use_stage1:
                            attack_loss = self.criterion.compute_loss_transegpgd_stage1(
                                logits_adv, patched_label, logits_clean)
                        else:
                            attack_loss = self.criterion.compute_loss_transegpgd_stage2_js(
                                logits_adv, patched_label, logits_clean)

                        with torch.no_grad():
                            patch_mask = self._estimate_patch_mask(image, patched_image)
                        
                        ah_loss = self.attn_hijack_loss(atts_adv, patch_mask, H, W)

                        if ah_loss.numel() == 0 or (ah_loss.detach().abs() < 1e-12):
                            # fallback: saliency hijack (model-agnostic)
                            # NOTE: saliency_hijack needs gradients through `patched_image`
                            patched_image.requires_grad_(True)
                            ah_loss = self.saliency_hijack_loss(logits_adv, patched_image, patch_mask)
                        tv = self.tv_loss(base_patch)
                        b_loss = self.boundary_disruption_loss(logits_clean, logits_adv)
                        f_loss = self.freq_shaping_loss(base_patch)

                        ga_loss = torch.zeros((), device=self.device)
                        use_sur_now = (self.use_surrogate and self.grad_align_w > 0.0 and
                                       (surrogate_every <= 1 or (it % surrogate_every == 0)))
                        if use_sur_now:
                            sur_logits = self.surrogate_forward_logits(patched_image, target_size=target_hw)
                            if sur_logits is not None:
                                ga_loss = self.logit_agreement_loss(logits_adv.detach(), sur_logits)

                        
                                # Option 2: KL alignment (comment out option 1 to use this)
                                # ga_loss = self.kl_align(logits_adv, sur_logits, T=1.0)


                        total_inner = (-attack_loss) \
                                      + self.tv_weight * tv \
                                      + self.attn_w * ah_loss \
                                      + self.boundary_w * b_loss \
                                      + self.freq_w * f_loss \
                                      + self.grad_align_w * ga_loss

                        self.model.zero_grad(set_to_none=True)
                        if self.surrogate is not None: self.surrogate.zero_grad(set_to_none=True)
                        if self.patch_param.grad is not None: self.patch_param.grad.zero_()
                        total_inner.backward()

                        with torch.no_grad():
                            self.patch_param += alpha * self.patch_param.grad.sign()
                            vis = self.get_patch().clamp(0,1)
                            z = (vis*2 - 1).clamp(-0.999, 0.999)
                            self.patch_param.copy_(torch.atanh(z))

                # --- metrics / logging ---
                self.metric.update(logits_adv, patched_label)
                _, mIoU = self.metric.get()
                cum_attack_loss += attack_loss.item()

                if it % self.log_every == 0:
                    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
                    eta_sec = ((time.time() - start_time) / max(1, (ep - self.start_epoch) * self.iters_per_epoch + it + 1)) * \
                              (self.total_epochs * self.iters_per_epoch - ((ep - self.start_epoch) * self.iters_per_epoch + it + 1))
                    eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                    lr_show = (self.opt.param_groups[0]["lr"] if self.opt is not None else 0.0)
                    self.log.info(
                        f"Epoch {ep}/{end_epoch} || Batch {it+1}/{self.iters_per_epoch} || "
                        f"LR: {lr_show:.3e} || "
                        f"Atk: {attack_loss.item():.4f} || "
                        f"Attn:{ah_loss.item():.4f} || TV:{tv.item():.4f} || "
                        f"Bound:{b_loss.item():.4f} || Freq:{f_loss.item():.4f} || "
                        f"GA:{ga_loss.item():.4f} || mIoU:{mIoU:.4f} || "
                        f"Elapsed:{elapsed} || ETA:{eta_str}"
                    )

            pixAcc, meanIoU = self.metric.get()
            avg_attack = cum_attack_loss / max(1, self.iters_per_epoch)
            self.log.info("-"*100)
            self.log.info(
                f"Epoch {ep}/{end_epoch} | {stage} | "
                f"Avg AttackLoss: {avg_attack:.4f} | Avg mIoU: {meanIoU:.4f} | Avg pixAcc: {pixAcc:.4f}"
            )
            self.log.info("-"*100)
            IoU_over_epochs.append(self.metric.get(full=True))

        return self.get_patch().detach(), np.array(IoU_over_epochs)
