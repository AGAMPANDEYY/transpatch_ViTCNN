# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from .arch import SegFormer_B

def load_segformer_local(variant_or_name: str,
                         ckpt_path: str,
                         device: torch.device,
                         num_classes: int = 19,
                         decoder_channels: int = 256,
                         strict: bool = False,
                         verbose: bool = True):
    """
    Load SegFormer (B0..B5) from a local .pth checkpoint produced by NVLabs/mmseg.
    """
    model = SegFormer_B(variant=variant_or_name.split('_')[-1] if 'segformer_' in variant_or_name else variant_or_name,
                        num_classes=num_classes,
                        decoder_channels=decoder_channels).to(device)
    model.eval()

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('state_dict', ckpt)
    new_state = {}
    for k, v in state.items():
        k2 = k
        if k2.startswith('module.'):
            k2 = k2[len('module.'):]
        if k2.startswith('model.'):
            k2 = k2[len('model.'):]
        if k2.startswith('auxiliary_head.'):
            continue
        # Map checkpoint keys to model keys
        if k2 == 'decode_head.conv_seg.weight':
            k2 = 'decode_head.linear_pred.weight'
        elif k2 == 'decode_head.conv_seg.bias':
            k2 = 'decode_head.linear_pred.bias'
        elif k2 == 'decode_head.linear_fuse.conv.weight':
            k2 = 'decode_head.linear_fuse.weight'
        elif k2 == 'decode_head.linear_fuse.bn.weight':
            k2 = 'decode_head.bn.weight'
        elif k2 == 'decode_head.linear_fuse.bn.bias':
            k2 = 'decode_head.bn.bias'
        elif k2 == 'decode_head.linear_fuse.bn.running_mean':
            k2 = 'decode_head.bn.running_mean'
        elif k2 == 'decode_head.linear_fuse.bn.running_var':
            k2 = 'decode_head.bn.running_var'
        elif k2 == 'decode_head.linear_fuse.bn.num_batches_tracked':
            continue  # Skip bookkeeping parameter
        new_state[k2] = v

    msg = model.load_state_dict(new_state, strict=strict)

    if verbose:
        missing = msg.missing_keys if hasattr(msg, 'missing_keys') else []
        unexpected = msg.unexpected_keys if hasattr(msg, 'unexpected_keys') else []
        matched = sum(1 for k in new_state.keys() if k not in unexpected)
        print(f"[SegFormer] matched {matched}/{len(new_state)} keys | missing {len(missing)} | unexpected {len(unexpected)}")
        if missing:
            print("  missing:", missing[:10], "..." if len(missing) > 10 else "")
        if unexpected:
            print("  unexpected:", unexpected[:10], "..." if len(unexpected) > 10 else "")
    return model

@torch.no_grad()
def segformer_logits(model: SegFormer_B, x: torch.Tensor):
    """
    Forward to 1/4-resolution logits. Upsample yourself if you need full-res.
    """
    return model(x)  # [B, num_classes, H/4, W/4]
