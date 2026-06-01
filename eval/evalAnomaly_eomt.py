# eval/evalAnomaly_eomt.py — EoMT-COCO anomaly evaluation
import os
import sys
import glob
import yaml
import torch
import random
import importlib
import warnings
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from torch.nn import functional as F
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.metrics import average_precision_score


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eomt'))

from ood_metrics import fpr_at_95_tpr
from methods import msp_anomaly_score, maxlogit_anomaly_score, entropy_anomaly_score

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Device 
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

# Transforms
input_transform = Compose([
    Resize((512, 1024), Image.BILINEAR),
    ToTensor(),
])
target_transform = Compose([
    Resize((512, 1024), Image.NEAREST),
])


#Loading EoMT model (based on inference.ipynb) 
def load_eomt(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
    
    img_size = config["model"]["init_args"].get("img_size", 640)
    encoder = encoder_cls(img_size=img_size, **encoder_cfg.get("init_args", {}))

    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
    network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    network = network_cls(
        masked_attn_enabled=False,
        num_classes=133,
        encoder=encoder,
        **network_kwargs,
    )

    lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)
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
            state_dict = torch.load(state_dict_path,
                                    map_location=f"cuda:{device}" if device.type == "cuda" else device,
                                    weights_only=True)
            model.load_state_dict(state_dict, strict=False)

    return model, img_size


#Convert EoMT output to pixel logits [C, H, W] 
def masks_to_pixel_logits(mask_logits, class_logits):
    """
    Implements: sum_i p_i(c) * m_i[h,w]  (from the notebook's formula)
    mask_logits:   [N, H, W]  raw mask logits
    class_logits:  [N, C]     class logits per query
    returns:       [C, H, W]  pixel-level logits
    """
    mask_probs  = mask_logits.sigmoid()         
    class_probs = class_logits.softmax(dim=-1)  
    pixel_logits = torch.einsum('nc,nhw->chw', class_probs, mask_probs)
    return pixel_logits  


def rba_anomaly_score(mask_logits, class_logits):
    mask_probs       = mask_logits.sigmoid()                    
    class_confidence = class_logits.softmax(-1).max(-1).values   
    weighted = class_confidence[:, None, None] * mask_probs      
    rba = 1.0 - weighted.max(dim=0).values                       
    return rba.cpu().numpy()


def main():
    parser = ArgumentParser()
    parser.add_argument('--input', required=True,
                        help="Glob e.g. '../Validation_Dataset/RoadAnomaly21/images/*.png'")
    parser.add_argument('--config', required=True,
                        help="Path to EoMT YAML config")
    parser.add_argument('--method', default='msp',
                        choices=['msp', 'maxlogit', 'entropy', 'rba'])
    args = parser.parse_args()

    print(f"Loading EoMT model...")
    model, img_size = load_eomt(args.config)
    print(f"Method: {args.method.upper()}")

    anomaly_score_list = []
    ood_gts_list = []

    for path in sorted(glob.glob(os.path.expanduser(args.input))):
        print(f"  {path}")
        img = Image.open(path).convert('RGB')
        imgs = [(input_transform(img) * 255).to(torch.uint8).to(device)]
        img_sizes = [imgs[0].shape[-2:]]

        with torch.no_grad():
            transformed = model.resize_and_pad_imgs_instance_panoptic(imgs)
            mask_logits_per_layer, class_logits_per_layer = model(transformed)

            mask_logits = F.interpolate(
                mask_logits_per_layer[-1],
                size=img_sizes[0],
                mode='bilinear'
            ).squeeze(0)
            class_logits = class_logits_per_layer[-1].squeeze(0) 

            mask_logits = model.revert_resize_and_pad_logits_instance_panoptic(
                mask_logits.unsqueeze(0), img_sizes
            )[0]

        if args.method == 'rba':
            anomaly_result = rba_anomaly_score(mask_logits, class_logits)
        else:
            pixel_logits = masks_to_pixel_logits(mask_logits, class_logits)
            logits_np = pixel_logits.cpu().numpy()
            if args.method == 'msp':
                anomaly_result = msp_anomaly_score(logits_np)
            elif args.method == 'maxlogit':
                anomaly_result = maxlogit_anomaly_score(logits_np)
            elif args.method == 'entropy':
                anomaly_result = entropy_anomaly_score(logits_np)


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

        if device.type == 'mps':
            torch.mps.empty_cache()

    ood_gts     = np.array(ood_gts_list)
    anom_scores = np.array(anomaly_score_list)

    ood_out   = anom_scores[ood_gts == 1]
    ind_out   = anom_scores[ood_gts == 0]
    val_out   = np.concatenate([ind_out, ood_out])
    val_label = np.concatenate([np.zeros(len(ind_out)), np.ones(len(ood_out))])

    auprc = average_precision_score(val_label, val_out)
    fpr   = fpr_at_95_tpr(val_out, val_label)

    print(f"\n=== EoMT-COCO [{args.method.upper()}] ===")
    print(f"AUPRC:   {auprc * 100:.2f}%")
    print(f"FPR@95:  {fpr   * 100:.2f}%")

    with open('results_eomt_coco.txt', 'a') as f:
        f.write(f"[{args.method.upper()}] AUPRC: {auprc*100:.2f}  FPR@95: {fpr*100:.2f}  | {args.input}\n")


if __name__ == '__main__':
    main()