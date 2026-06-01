import os
import sys
import yaml
import torch
import numpy as np
import importlib
from PIL import Image
from argparse import ArgumentParser
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from torchmetrics.classification import MulticlassJaccardIndex

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eomt'))

from dataset import cityscapes
from transform import ToLabel, Relabel
from evalAnomaly_eomt import load_eomt, masks_to_pixel_logits

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

raw_to_train_array = np.full(256, 19, dtype=np.int64)
raw_to_train_array[7] = 0
raw_to_train_array[8] = 1
raw_to_train_array[11] = 2
raw_to_train_array[12] = 3
raw_to_train_array[13] = 4
raw_to_train_array[17] = 5
raw_to_train_array[19] = 6
raw_to_train_array[20] = 7
raw_to_train_array[21] = 8
raw_to_train_array[22] = 9
raw_to_train_array[23] = 10
raw_to_train_array[24] = 11
raw_to_train_array[25] = 12
raw_to_train_array[26] = 13
raw_to_train_array[27] = 14
raw_to_train_array[28] = 15
raw_to_train_array[29] = 16
raw_to_train_array[32] = 17
raw_to_train_array[33] = 18

COCO_TO_CITYSCAPES_MAP = {
    0: 11,
    1: 18,
    2: 13,
    3: 17,
    5: 15,
    6: 16,
    7: 14,
    115: 0,
    116: 1,
    129: 2,
    130: 3,
    131: 8,
    125: 10,
}

mapping_array = np.full(134, 19, dtype=np.int64)
for coco_id, city_id in COCO_TO_CITYSCAPES_MAP.items():
    mapping_array[coco_id] = city_id


def main(args):
    print(f"Loading EoMT Model with config: {args.config}")
    model, img_size = load_eomt(args.config)
    model.eval()

    input_transform = Compose([Resize((512, 1024), Image.BILINEAR), ToTensor()])
    target_transform = Compose([Resize((512, 1024), Image.NEAREST), ToLabel(), Relabel(255, 19)])

    loader = DataLoader(
        cityscapes(args.datadir, input_transform, target_transform, subset='val'),
        batch_size=1, shuffle=False, num_workers=2
    )

    iou_metric = MulticlassJaccardIndex(num_classes=20, ignore_index=19).to(device)

    print("Starting In-Distribution Validation Loop...")
    with torch.no_grad():
        for step, (images, raw_labels, _, _) in enumerate(loader):
            images = images.to(device)
            
            labels_np = raw_labels.squeeze(1).numpy()
            labels = torch.from_numpy(raw_to_train_array[labels_np]).to(device)

            imgs_uint8 = [(images[0] * 255).to(torch.uint8)]
            img_sizes = [imgs_uint8[0].shape[-2:]]

            transformed = model.resize_and_pad_imgs_instance_panoptic(imgs_uint8)
            mask_logits_layer, class_logits_layer = model(transformed)

            mask_logits = F.interpolate(mask_logits_layer[-1], size=img_sizes[0], mode='bilinear').squeeze(0)
            class_logits = class_logits_layer[-1].squeeze(0)
            mask_logits = model.revert_resize_and_pad_logits_instance_panoptic(mask_logits.unsqueeze(0), img_sizes)[0]

            pixel_logits = masks_to_pixel_logits(mask_logits, class_logits)

            coco_pred = torch.argmax(pixel_logits, dim=0).cpu().numpy()
            city_pred = torch.from_numpy(mapping_array[coco_pred]).to(device).unsqueeze(0)

            iou_metric.update(city_pred, labels)

            if step % 50 == 0:
                print(f"  Processed {step}/{len(loader)} frames...")

    final_miou = iou_metric.compute()
    print(f"\n=== CITYSCAPES VAL RESULTS (EoMT COCO) ===")
    print(f"Semantic mIoU: {final_miou.item() * 100:.2f}%")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', default='eomt/configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml')
    parser.add_argument('--datadir', required=True, help="Path to your local cityscapes dataset folder")
    main(parser.parse_args())