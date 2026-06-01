import os
import sys
import glob
import random
import torch
import torch.nn.functional as F  # type: ignore[import]
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.metrics import average_precision_score

# ---------------------------------------------------------------------------
# EoMT imports — add eomt/ to sys.path so models are importable without
# installing the package. Works regardless of the cwd this script is run from.
# ---------------------------------------------------------------------------
_EOMT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'eomt')
sys.path.insert(0, _EOMT_DIR)

from models.eomt import EoMT   # type: ignore[import]  # noqa: E402
from models.vit import ViT     # type: ignore[import]  # noqa: E402

# ---------------------------------------------------------------------------
# Local imports from the same eval/ folder
# ---------------------------------------------------------------------------
from ood_metrics import fpr_at_95_tpr                          # noqa: E402
from methods import (                                           # noqa: E402
    msp_anomaly_score,
    maxlogit_anomaly_score,
    entropy_anomaly_score,
)

# ---------------------------------------------------------------------------
# Reproducibility (matches evalAnomaly.py)
# ---------------------------------------------------------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# ---------------------------------------------------------------------------
# Constants — must match the Cityscapes-semantic checkpoint.
# Source: eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml
# ---------------------------------------------------------------------------
NUM_CLASSES   = 19           # Cityscapes classes (void handled separately in class_head)
NUM_QUERIES   = 100          # num_q in config
NUM_BLOCKS    = 3            # num_blocks in config
BACKBONE_NAME = "vit_base_patch14_reg4_dinov2"
TRAIN_IMG_SIZE = (1024, 1024)  # derived from CityscapesSemantic.img_size default (no override in YAML);
                                # linked via main.py:133 to encoder.init_args.img_size
EVAL_IMG_SIZE  = (1024, 1024) # anomaly dataset inference resolution

# ---------------------------------------------------------------------------
# Transforms (identical to evalAnomaly.py).
# ToTensor() → [0, 1] float; EoMT normalises internally via pixel_mean/pixel_std.
# ---------------------------------------------------------------------------
input_transform = Compose([
    Resize(EVAL_IMG_SIZE, Image.BILINEAR),
    ToTensor(),
])

target_transform = Compose([
    Resize(EVAL_IMG_SIZE, Image.NEAREST),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_device(cpu_flag: bool) -> torch.device:
    if cpu_flag:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_eomt() -> EoMT:
    """
    Build the EoMT-base skeleton that matches the Cityscapes checkpoint.

    Parameters are derived from the YAML config + Python defaults — NOT guessed:
      img_size   = (1024, 1024)  CityscapesSemantic default; linked to ViT via main.py
      patch_size = 16            ViT.__init__ default; YAML has no patch_size key
      backbone   = vit_base_patch14_reg4_dinov2  (just the pretrained name; patch14
                                  refers to the original DINOv2 pretraining, not what
                                  is used here — timm rebuilds patch_embed with patch_size=16)

    ckpt_path='placeholder' sets pretrained=False in timm, skipping the DINOv2
    download. All weights come from the checkpoint loaded in load_eomt_checkpoint().
    """
    encoder = ViT(
        img_size=TRAIN_IMG_SIZE,   # (1024, 1024)
        patch_size=16,             # explicit — ViT default, not overridden in YAML
        backbone_name=BACKBONE_NAME,
        ckpt_path='placeholder',
    )
    return EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=NUM_QUERIES,
        num_blocks=NUM_BLOCKS,
    )


def load_eomt_checkpoint(model: EoMT, ckpt_path: str) -> EoMT:
    """
    Load an EoMT-Cityscapes checkpoint into the model.

    Handles the formats produced by LightningModule._load_ckpt / on_save_checkpoint:
      - Flat state dict with 'network.' prefix  (pytorch_model.bin style)
      - Nested Lightning .ckpt with 'state_dict' key
    Filters 'criterion.*' and 'metrics.*' keys which are not part of EoMT.
    """
    print(f"Loading checkpoint: {ckpt_path}")

    # Match _load_ckpt in lightning_module.py (weights_only=True is safe default).
    # Fall back to False for checkpoints saved with older PyTorch.
    try:
        raw = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    except Exception:
        print("  weights_only=True failed — retrying with weights_only=False.")
        raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Unwrap nested Lightning .ckpt format
    if isinstance(raw, dict) and 'state_dict' in raw:
        print("  Lightning .ckpt detected — extracting state_dict.")
        raw = raw['state_dict']

    # Drop keys that don't belong in EoMT at inference time
    raw = {k: v for k, v in raw.items()
           if not k.startswith('criterion.')
           and not k.startswith('metrics.')}

    # Strip the 'network.' prefix added by MaskClassificationSemantic
    # (which stores EoMT as self.network).
    sample = list(raw.keys())[:5]
    if sample and all(k.startswith('network.') for k in sample):
        print("  Stripping 'network.' prefix from keys.")
        raw = {k[len('network.'):]: v for k, v in raw.items()}

    result = model.load_state_dict(raw, strict=False)

    # attn_mask_probs is a training buffer registered in EoMT; it should be
    # in the checkpoint and load normally. Only warn about truly missing keys.
    real_missing = [k for k in result.missing_keys if 'attn_mask_probs' not in k]
    if real_missing:
        print(f"  WARNING — {len(real_missing)} missing keys "
              f"(first 5): {real_missing[:5]}")
    else:
        print("  Checkpoint loaded successfully.")

    return model


def eomt_to_pixel_logits(
    mask_logits_per_layer: list,
    class_logits_per_layer: list,
    target_hw: tuple,
) -> torch.Tensor:
    """
    Convert EoMT output into per-pixel class logits for anomaly scoring.

    Uses the last layer only (best quality prediction).

    Mirrors LightningModule.to_per_pixel_logits_semantic (lightning_module.py:668):
        pixel_logits = einsum("bqhw, bqc -> bchw",
                              mask_logits.sigmoid(),
                              class_logits.softmax(dim=-1)[..., :-1])

    [..:-1] removes the void/no-object class (class_head outputs NUM_CLASSES+1).
    mask_logits is resized to target_hw before the einsum, matching the official
    eval_step which does F.interpolate before calling to_per_pixel_logits_semantic.

    Returns:
        pixel_logits: [B, NUM_CLASSES, H, W] float tensor
    """
    mask_logits  = mask_logits_per_layer[-1]    # [B, Q, H', W']
    class_logits = class_logits_per_layer[-1]   # [B, Q, C+1]

    # Resize masks before combining — cheaper than resizing pixel_logits
    # (100 query channels vs 19 class channels).
    mask_logits_r = F.interpolate(
        mask_logits,
        size=target_hw,
        mode='bilinear',
        align_corners=False,
    )  # [B, Q, H, W]

    pixel_logits = torch.einsum(
        "bqhw, bqc -> bchw",
        mask_logits_r.sigmoid(),
        class_logits.softmax(dim=-1)[..., :-1],
    )  # [B, C, H, W]

    return pixel_logits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = ArgumentParser(
        description="Anomaly segmentation evaluation for EoMT-Cityscapes. "
                    "Run from the eval/ directory with the eomt conda env active."
    )
    parser.add_argument(
        '--input',
        default='/path/to/RoadAnomaly21/images/*.png',
        nargs='+',
        help="Glob pattern for input images, "
             "e.g. '/data/RoadAnomaly21/images/*.png'",
    )
    parser.add_argument(
        '--checkpoint',
        default='../eomt/checkpoints/eomt_cityscapes.bin',
        help="Path to the EoMT Cityscapes checkpoint (.bin / .ckpt / .pth)",
    )
    parser.add_argument(
        '--method',
        default='maxlogit',
        choices=['msp', 'maxlogit', 'entropy'],
        help="Anomaly scoring method: msp | maxlogit | entropy",
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help="Force CPU inference (avoids MPS/CUDA issues if needed)",
    )
    args = parser.parse_args()

    method_map = {
        'msp':      msp_anomaly_score,
        'maxlogit': maxlogit_anomaly_score,
        'entropy':  entropy_anomaly_score,
    }
    score_fn = method_map[args.method]
    print(f"Anomaly scoring method : {args.method.upper()}")

    device = get_device(args.cpu)
    print(f"Device                 : {device}")

    # --- Build and load model ---
    print("Building EoMT-Cityscapes model skeleton...")
    model = build_eomt()
    model = load_eomt_checkpoint(model, args.checkpoint)
    model = model.to(device)
    model.eval()

    # --- Separate results file (does not overwrite ERFNet results.txt) ---
    results_path = 'results_eomt.txt'
    if not os.path.exists(results_path):
        open(results_path, 'w').close()
    results_file = open(results_path, 'a')

    anomaly_score_list = []
    ood_gts_list = []

    # --- Inference loop ---
    image_paths = sorted(glob.glob(os.path.expanduser(str(args.input[0]))))
    if not image_paths:
        print(f"\nERROR: No images matched: {args.input[0]}")
        print("  Make sure to quote the glob pattern, e.g.:")
        print("  python evalAnomaly_eomt.py --input '/data/RoadAnomaly21/images/*.png'")
        results_file.close()
        return

    print(f"Found {len(image_paths)} images. Starting inference...")

    for path in image_paths:
        print(path)

        img = Image.open(path).convert('RGB')
        tensor_img = input_transform(img).unsqueeze(0).float().to(device)

        with torch.no_grad():
            mask_logits_per_layer, class_logits_per_layer = model(tensor_img)

        pixel_logits = eomt_to_pixel_logits(
            mask_logits_per_layer,
            class_logits_per_layer,
            target_hw=EVAL_IMG_SIZE,
        )  # [1, 19, 512, 1024]

        logits_np = pixel_logits.squeeze(0).cpu().numpy()  # [19, 512, 1024]
        anomaly_result = score_fn(logits_np)                # [512, 1024]

        # --- GT mask loading (identical to evalAnomaly.py) ---
        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts == 2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts == 0), 255, ood_gts)
            ood_gts = np.where((ood_gts == 1), 0, ood_gts)
            ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)
        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts == 14), 255, ood_gts)
            ood_gts = np.where((ood_gts < 20), 0, ood_gts)
            ood_gts = np.where((ood_gts == 255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue

        ood_gts_list.append(ood_gts)
        anomaly_score_list.append(anomaly_result)

        del mask_logits_per_layer, class_logits_per_layer, pixel_logits
        del anomaly_result, ood_gts, mask

        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()

    if not anomaly_score_list:
        print("\nERROR: No valid images were processed.")
        print("  Check that labels_masks/ exists next to images/ with matching filenames.")
        results_file.close()
        return

    # --- Metrics (identical to evalAnomaly.py) ---
    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out   = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr     = fpr_at_95_tpr(val_out, val_label)

    print(f'\nAUPRC score : {prc_auc * 100.0:.4f}')
    print(f'FPR@TPR95   : {fpr * 100.0:.4f}')

    results_file.write('\n')
    results_file.write(
        f'    [EoMT-Cityscapes][{args.method.upper()}]'
        f'  AUPRC score:{prc_auc * 100.0}'
        f'   FPR@TPR95:{fpr * 100.0}'
    )
    results_file.close()


if __name__ == '__main__':
    main()
