import os
import sys
import glob
import yaml
import random
import importlib
import warnings
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.metrics import average_precision_score

# EoMT repo on path
_EOMT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'eomt')
sys.path.insert(0, _EOMT_DIR)

from models.eomt import EoMT
from models.vit import ViT
from ood_metrics import fpr_at_95_tpr 
from methods import (                 
    msp_anomaly_score,
    maxlogit_anomaly_score,
    entropy_anomaly_score,
)

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Cityscapes-specific constants
CS_NUM_CLASSES    = 19
CS_NUM_QUERIES    = 100
CS_NUM_BLOCKS     = 3
CS_BACKBONE_NAME  = "vit_base_patch14_reg4_dinov2"
CS_TRAIN_IMG_SIZE = (1024, 1024)
CS_EVAL_IMG_SIZE  = (1024, 1024)

# Shared transforms — each model loader overrides eval size if needed
def make_transforms(eval_size):
    return (
        Compose([Resize(eval_size, Image.BILINEAR), ToTensor()]),
        Compose([Resize(eval_size, Image.NEAREST)]),
    )

# Device helper
def get_device(cpu_flag: bool) -> torch.device:
    if cpu_flag:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

# COCO model loader  (Person 2)
def load_eomt_coco(config_path, device):
    """Load EoMT pretrained on COCO panoptic via HuggingFace Hub."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)

    img_size = config["model"]["init_args"].get("img_size", 640)
    encoder  = encoder_cls(img_size=img_size, **encoder_cfg.get("init_args", {}))

    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls    = getattr(importlib.import_module(network_module_name), network_class_name)
    network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    network = network_cls(
        masked_attn_enabled=False,
        num_classes=133,
        encoder=encoder,
        **network_kwargs,
    )

    lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls     = getattr(importlib.import_module(lit_module_name), lit_class_name)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}

    if "Panoptic" in lit_class_name:
        model_kwargs["stuff_classes"] = list(range(80, 133))

    model = (
        lit_cls(
            img_size=(img_size, img_size),
            num_classes=133,
            network=network,
            **model_kwargs,
        )
        .eval()
        .to(torch.float32)
        .to(device)
    )

    from huggingface_hub import hf_hub_download
    name = config.get("trainer", {}).get("logger", {}).get("init_args", {}).get("name")
    if name == "coco_panoptic_eomt_base_640":
        name = "coco_panoptic_eomt_base_640_2x"
    if name:
        is_dinov3 = "dinov3" in name
        state_dict_path = hf_hub_download(
            repo_id=f"tue-mps/{name}",
            filename="pytorch_model.bin",
        )
        if is_dinov3:
            model_kwargs["ckpt_path"] = state_dict_path
            model_kwargs["delta_weights"] = True
        else:
            state_dict = torch.load(
                state_dict_path,
                map_location=f"cuda:{device}" if device.type == "cuda" else device,
                weights_only=True,
            )
            model.load_state_dict(state_dict, strict=False)

    return model, img_size


# Cityscapes model loader  (Person 3)
def build_eomt_cityscapes() -> EoMT:
    encoder = ViT(
        img_size=CS_TRAIN_IMG_SIZE,
        patch_size=16,
        backbone_name=CS_BACKBONE_NAME,
        ckpt_path='placeholder',
    )
    return EoMT(
        encoder=encoder,
        num_classes=CS_NUM_CLASSES,
        num_q=CS_NUM_QUERIES,
        num_blocks=CS_NUM_BLOCKS,
    )


def load_eomt_cityscapes(ckpt_path: str, device: torch.device) -> EoMT:
    model = build_eomt_cityscapes()
    print(f"Loading Cityscapes checkpoint: {ckpt_path}")
    try:
        raw = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    except Exception:
        print("  weights_only=True failed — retrying with weights_only=False.")
        raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    if isinstance(raw, dict) and 'state_dict' in raw:
        print("  Lightning .ckpt detected — extracting state_dict.")
        raw = raw['state_dict']

    raw = {k: v for k, v in raw.items()
           if not k.startswith('criterion.') and not k.startswith('metrics.')}

    sample = list(raw.keys())[:5]
    if sample and all(k.startswith('network.') for k in sample):
        print("  Stripping 'network.' prefix.")
        raw = {k[len('network.'):]: v for k, v in raw.items()}

    result = model.load_state_dict(raw, strict=False)
    real_missing = [k for k in result.missing_keys if 'attn_mask_probs' not in k]
    if real_missing:
        print(f"  WARNING — {len(real_missing)} missing keys (first 5): {real_missing[:5]}")
    else:
        print("  Checkpoint loaded successfully.")

    return model.to(device).eval()


# Shared projection helpers
def coco_masks_to_pixel_logits(mask_logits, class_logits):
    """
    Person 2 projection for COCO model (LightningModule wrapper).
    mask_logits:  [N, H, W]  raw mask logits
    class_logits: [N, C]     class logits per query
    returns:      [C, H, W]  pixel-level logits
    """
    mask_probs   = mask_logits.sigmoid()
    class_probs  = class_logits.softmax(dim=-1)
    return torch.einsum('nc,nhw->chw', class_probs, mask_probs)


def cityscapes_masks_to_pixel_logits(mask_logits_per_layer, class_logits_per_layer, target_hw):
    """
    Person 3 projection for Cityscapes model (raw EoMT).
    Mirrors LightningModule.to_per_pixel_logits_semantic.
    Returns: [B, C, H, W]
    """
    mask_logits  = mask_logits_per_layer[-1]
    class_logits = class_logits_per_layer[-1]

    mask_logits_r = F.interpolate(
        mask_logits, size=target_hw, mode='bilinear', align_corners=False
    )
    return torch.einsum(
        "bqhw, bqc -> bchw",
        mask_logits_r.sigmoid(),
        class_logits.softmax(dim=-1)[..., :-1],
    )


def rba_anomaly_score_coco(mask_logits, class_logits):
    """
    RbA for the COCO model (LightningModule panoptic output).
    mask_logits:  [N, H, W]  raw mask logits (already resized + un-padded)
    class_logits: [N, C]     class logits per query
    A pixel is anomalous if no mask confidently claims it.
    Score = 1 - max_n( sigmoid(mask_n[h,w]) * max_c softmax(class_n) )
    """
    mask_probs       = mask_logits.sigmoid()                     # [N, H, W]
    class_confidence = class_logits.softmax(-1).max(-1).values   # [N]
    weighted         = class_confidence[:, None, None] * mask_probs  # [N, H, W]
    return (1.0 - weighted.max(dim=0).values).cpu().numpy()      # [H, W]


def rba_anomaly_score_cityscapes(mask_logits_per_layer, class_logits_per_layer, target_hw):
    """
    RbA for the Cityscapes model (raw EoMT output).
    Uses the last decoder layer only (best quality).

    The Cityscapes class head outputs C+1 logits (last = void/no-object class).
    We exclude the void class from the confidence calculation — a mask that
    confidently predicts 'void' should NOT suppress the anomaly score, because
    void predictions are exactly what happens on OoD pixels.

    Score = 1 - max_n( sigmoid(mask_n[h,w]) * max_{c != void} softmax(class_n)[c] )
    """
    mask_logits  = mask_logits_per_layer[-1]   # [B, Q, H', W']
    class_logits = class_logits_per_layer[-1]  # [B, Q, C+1]

    # Resize masks to target resolution before combining
    mask_logits_r = F.interpolate(
        mask_logits, size=target_hw, mode='bilinear', align_corners=False
    ).squeeze(0)  # [Q, H, W]

    # Exclude void class ([..., :-1]) then take max in-class confidence per query
    class_probs      = class_logits.softmax(dim=-1)[..., :-1]  # [B, Q, C]
    class_confidence = class_probs.squeeze(0).max(dim=-1).values  # [Q]

    weighted = class_confidence[:, None, None] * mask_logits_r.sigmoid()  # [Q, H, W]
    return (1.0 - weighted.max(dim=0).values).cpu().numpy()               # [H, W]


# GT mask loading
def load_gt_mask(path, target_transform):
    pathGT = path.replace("images", "labels_masks")
    if "RoadObsticle21" in pathGT:
        pathGT = pathGT.replace("webp", "png")
    if "fs_static" in pathGT:
        pathGT = pathGT.replace("jpg", "png")
    if "RoadAnomaly" in pathGT:
        pathGT = pathGT.replace("jpg", "png")

    mask    = Image.open(pathGT)
    mask    = target_transform(mask)
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

    return ood_gts


# Main
def main():
    parser = ArgumentParser(
        description="Unified EoMT anomaly evaluation (COCO or Cityscapes)."
    )
    parser.add_argument('--model', required=True, choices=['coco', 'cityscapes'],
                        help="Which EoMT checkpoint to use")
    # COCO-specific
    parser.add_argument('--config', default=None,
                        help="[coco only] Path to EoMT YAML config")
    # Cityscapes-specific
    parser.add_argument('--checkpoint', default=None,
                        help="[cityscapes only] Path to .bin/.ckpt checkpoint")
    # Shared
    parser.add_argument('--input', required=True,
                        help="Glob for input images, e.g. '...RoadAnomaly21/images/*.png'")
    parser.add_argument('--method', default='maxlogit',
                        choices=['msp', 'maxlogit', 'entropy', 'rba'],
                        help="Anomaly scoring method (rba: coco only)")
    parser.add_argument('--cpu', action='store_true', help="Force CPU inference")
    args = parser.parse_args()

    # Validate argument combinations
    if args.model == 'coco' and args.config is None:
        parser.error("--config is required when --model coco")
    if args.model == 'cityscapes' and args.checkpoint is None:
        parser.error("--checkpoint is required when --model cityscapes")
    device = get_device(args.cpu)
    print(f"Model   : EoMT-{args.model.upper()}")
    print(f"Method  : {args.method.upper()}")
    print(f"Device  : {device}")

    # Load model + set up transforms
    if args.model == 'coco':
        os.makedirs('saved_logits_eomt_coco', exist_ok=True)
        model, img_size = load_eomt_coco(args.config, device)
        eval_size       = (img_size, img_size)
        results_file_path = 'results_eomt_coco.txt'
    else:
        model     = load_eomt_cityscapes(args.checkpoint, device)
        eval_size = CS_EVAL_IMG_SIZE
        results_file_path = 'results_eomt_cityscapes.txt'

    input_transform, target_transform = make_transforms(eval_size)

    results_file = open(results_file_path, 'a')

    # Inference loop
    image_paths = sorted(glob.glob(os.path.expanduser(args.input)))
    if not image_paths:
        print(f"\nERROR: No images matched: {args.input}")
        results_file.close()
        return
    print(f"Found {len(image_paths)} images.")

    anomaly_score_list = []
    ood_gts_list       = []

    for path in image_paths:
        print(f"  {path}")
        img = Image.open(path).convert('RGB')

        if args.model == 'coco':
            imgs      = [(input_transform(img) * 255).to(torch.uint8).to(device)]
            img_sizes = [imgs[0].shape[-2:]]
            with torch.no_grad():
                transformed = model.resize_and_pad_imgs_instance_panoptic(imgs)
                mask_logits_per_layer, class_logits_per_layer = model(transformed)
                mask_logits  = F.interpolate(
                    mask_logits_per_layer[-1], size=img_sizes[0], mode='bilinear'
                ).squeeze(0)
                class_logits = class_logits_per_layer[-1].squeeze(0)
                mask_logits  = model.revert_resize_and_pad_logits_instance_panoptic(
                    mask_logits.unsqueeze(0), img_sizes
                )[0]

            # Save logits (temperature scaling)
            dataset_name = path.split('/')[-3]
            base_name = os.path.basename(path).split('.')[0]
            pixel_logits_save = coco_masks_to_pixel_logits(mask_logits, class_logits)
            torch.save(pixel_logits_save.cpu(),
                       os.path.join('saved_logits_eomt_coco', f"{dataset_name}_{base_name}_logits.pt"))

            if args.method == 'rba':
                anomaly_result = rba_anomaly_score_coco(mask_logits, class_logits)
            else:
                pixel_logits   = coco_masks_to_pixel_logits(mask_logits, class_logits)
                logits_np      = pixel_logits.cpu().numpy()
                anomaly_result = {
                    'msp':      msp_anomaly_score,
                    'maxlogit': maxlogit_anomaly_score,
                    'entropy':  entropy_anomaly_score,
                }[args.method](logits_np)

        else:  # cityscapes
            tensor_img = input_transform(img).unsqueeze(0).float().to(device)
            with torch.no_grad():
                mask_logits_per_layer, class_logits_per_layer = model(tensor_img)

            if args.method == 'rba':
                anomaly_result = rba_anomaly_score_cityscapes(
                    mask_logits_per_layer, class_logits_per_layer, target_hw=eval_size
                )
            else:
                pixel_logits   = cityscapes_masks_to_pixel_logits(
                    mask_logits_per_layer, class_logits_per_layer, target_hw=eval_size
                )
                logits_np      = pixel_logits.squeeze(0).cpu().numpy()
                anomaly_result = {
                    'msp':      msp_anomaly_score,
                    'maxlogit': maxlogit_anomaly_score,
                    'entropy':  entropy_anomaly_score,
                }[args.method](logits_np)

        # GT mask
        ood_gts = load_gt_mask(path, target_transform)
        if 1 not in np.unique(ood_gts):
            continue

        ood_gts_list.append(ood_gts)
        anomaly_score_list.append(anomaly_result)

        # Memory cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()

    if not anomaly_score_list:
        print("\nERROR: No valid images processed.")
        results_file.close()
        return

    # Metrics
    ood_gts        = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_out   = anomaly_scores[ood_gts == 1]
    ind_out   = anomaly_scores[ood_gts == 0]
    val_out   = np.concatenate([ind_out, ood_out])
    val_label = np.concatenate([np.zeros(len(ind_out)), np.ones(len(ood_out))])

    auprc = average_precision_score(val_label, val_out)
    fpr   = fpr_at_95_tpr(val_out, val_label)

    tag = f"EoMT-{args.model.upper()}"
    print(f"\n=== {tag} [{args.method.upper()}] ===")
    print(f"AUPRC  : {auprc * 100:.2f}%")
    print(f"FPR@95 : {fpr   * 100:.2f}%")

    results_file.write(
        f"[{args.method.upper()}] AUPRC: {auprc*100:.2f}  FPR@95: {fpr*100:.2f}"
        f"  | {args.input}\n"
    )
    results_file.close()


if __name__ == '__main__':
    main()