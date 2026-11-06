import torch
import torchvision.models
from torchvision.datasets import ImageNet
import torch.nn as nn
from torchvision import transforms as transforms
from tqdm import tqdm
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import os
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path
import numpy as np

device = torch.device("cuda")


def save_inference_cache(cache, save_path='inference_cache.pkl'):
    with open(save_path, 'wb') as f:
        pickle.dump(cache, f)
    print(f"Cache saved to {save_path}")


def load_inference_cache(load_path='inference_cache.pkl'):
    if Path(load_path).exists():
        with open(load_path, 'rb') as f:
            cache = pickle.load(f)
        print(f"Cache loaded from {load_path}")
        return cache
    else:
        print(f"Cache file not found: {load_path}")
        return None


def parse_bbox_annotation(xml_path):
    if not os.path.exists(xml_path):
        return []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        bboxes.append([xmin, ymin, xmax, ymax])
    return bboxes


class MaskedImageDataset(Dataset):
    def __init__(self, data_list, transform):
        self.data_list = data_list
        self.transform = transform
        self.preprocess()

    def preprocess(self):
        print("Preprocessing bboxes...")
        for data in tqdm(self.data_list):
            img = Image.open(data['image_path'])
            orig_w, orig_h = img.size
            bboxes = parse_bbox_annotation(data['xml_path'])
            data['orig_size'] = (orig_w, orig_h)
            data['bboxes'] = bboxes

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        img = Image.open(data['image_path']).convert('RGB')
        orig_w, orig_h = data['orig_size']
        bboxes = data['bboxes']

        img_tensor = self.transform(img)

        bbox_mask_pil = Image.new('L', (orig_w, orig_h), 0)
        draw = ImageDraw.Draw(bbox_mask_pil)
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox
            draw.rectangle([xmin, ymin, xmax, ymax], fill=255)

        mask_transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        bbox_mask_tensor = mask_transform(bbox_mask_pil)
        bbox_mask_tensor = (bbox_mask_tensor > 0.5).float()
        mask = 1 - bbox_mask_tensor.repeat(3, 1, 1)

        object_tensor = img_tensor * (1 - mask)
        background_tensor = img_tensor * mask

        return img_tensor, object_tensor, background_tensor, data['class'], mask, idx


def load_model(model, path):
    if os.path.isfile(path):
        datas = torch.load(path, weights_only=False)
        if 'model' in datas:
            state_dict = {k.replace('module.', '').replace('backbone.', ''): v
                          for k, v in datas['model'].items()
                          if k.replace('module.', '').startswith('backbone.')}
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(datas)
        print(f"loaded state dict {path}")


def extract_train_features(model, train_loader, num_samples=None):
    """Extract features from training set for kNN."""
    print("Extracting train features...")
    model.eval()
    all_features = []
    all_labels = []
    count = 0

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for images, labels in tqdm(train_loader, desc="Train features"):
            images = images.to(device, non_blocking=True)
            features = model(images)
            # L2 normalize
            features = nn.functional.normalize(features, dim=1)
            all_features.append(features.cpu())
            all_labels.append(labels)
            count += images.size(0)
            if num_samples is not None and count >= num_samples:
                break

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    print(f"Extracted {all_features.shape[0]} train features, dim={all_features.shape[1]}")
    return all_features, all_labels


def knn_classify(train_features, train_labels, query_features, k=20, T=0.07, num_classes=1000):
    """
    kNN classification.
    Returns per-sample: predicted class, top-1 correct, top-5 correct, confidence for true class.
    """
    # query_features: (N, D), train_features: (M, D)
    # Process in chunks to avoid OOM
    chunk_size = 256
    N = query_features.shape[0]

    all_top1 = []
    all_top5 = []
    all_conf = []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        query_chunk = query_features[start:end].to(device)  # (chunk, D)

        # Cosine similarity (features are already L2 normalized)
        sim = torch.mm(query_chunk, train_features.t().to(device))  # (chunk, M)

        # Top-k neighbors
        topk_sim, topk_idx = sim.topk(k, dim=1)  # (chunk, k)
        topk_labels = train_labels.to(device)[topk_idx]  # (chunk, k)

        # Weighted voting with temperature
        weights = torch.exp(topk_sim / T)  # (chunk, k)

        # Accumulate votes per class
        votes = torch.zeros(end - start, num_classes, device=device)
        votes.scatter_add_(1, topk_labels, weights)

        # Normalize to get probabilities
        probs = votes / votes.sum(dim=1, keepdim=True)

        # Top-1 and Top-5 predictions
        top5_preds = probs.topk(5, dim=1)[1]  # (chunk, 5)
        top1_preds = top5_preds[:, 0]

        all_top1.append(top1_preds.cpu())
        all_top5.append(top5_preds.cpu())
        all_conf.append(probs.cpu())

    all_top1 = torch.cat(all_top1, dim=0)
    all_top5 = torch.cat(all_top5, dim=0)
    all_conf = torch.cat(all_conf, dim=0)

    return all_top1, all_top5, all_conf


def knn_inference(steps, models_steps, save_path, masking_dataset, masking_loader,
                  train_loader, k=20, T=0.07):
    """Run kNN-based inference instead of linear probing."""
    inference_cache = {step: {} for step in steps}

    for step, model in zip(steps, models_steps):
        print(f"\n{'=' * 60}")
        print(f"  [Step {step}] kNN Inference (k={k}, T={T})")
        print(f"{'=' * 60}")
        model.to(device).eval()

        # 1. Extract train features
        train_features, train_labels = extract_train_features(model, train_loader)

        # 2. Extract val features (orig, obj, bg) and run kNN
        print("Extracting val features and running kNN...")
        list_classes = []
        list_orig_top1 = []
        list_orig_top5 = []
        list_obj_top1 = []
        list_obj_top5 = []
        list_bg_top1 = []
        list_bg_top5 = []
        list_conf_orig = []
        list_conf_obj = []
        list_conf_bg = []

        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            all_orig_feats = []
            all_obj_feats = []
            all_bg_feats = []
            all_classes = []

            for orig_batch, obj_batch, bg_batch, classes, _, indices in tqdm(masking_loader, desc=f"Features {step}"):
                B = orig_batch.size(0)

                combined = torch.cat([orig_batch, obj_batch, bg_batch], dim=0).to(device, non_blocking=True)
                features = model(combined)
                features = nn.functional.normalize(features, dim=1)

                orig_feats, obj_feats, bg_feats = features.split(B, dim=0)

                all_orig_feats.append(orig_feats.cpu())
                all_obj_feats.append(obj_feats.cpu())
                all_bg_feats.append(bg_feats.cpu())
                all_classes.append(classes)

        all_orig_feats = torch.cat(all_orig_feats, dim=0)
        all_obj_feats = torch.cat(all_obj_feats, dim=0)
        all_bg_feats = torch.cat(all_bg_feats, dim=0)
        all_classes = torch.cat(all_classes, dim=0)

        # 3. kNN classify each condition
        print("kNN classifying orig...")
        orig_top1, orig_top5, orig_conf = knn_classify(train_features, train_labels, all_orig_feats, k=k, T=T)
        print("kNN classifying obj...")
        obj_top1, obj_top5, obj_conf = knn_classify(train_features, train_labels, all_obj_feats, k=k, T=T)
        print("kNN classifying bg...")
        bg_top1, bg_top5, bg_conf = knn_classify(train_features, train_labels, all_bg_feats, k=k, T=T)

        # 4. Compute per-sample metrics
        N = len(masking_dataset)
        cache = inference_cache[step]

        for i in range(N):
            cls = all_classes[i].item()

            cache[i] = {
                'class': cls,
                'orig_top1': bool(orig_top1[i] == cls),
                'orig_top5': bool(cls in orig_top5[i]),
                'obj_top1': bool(obj_top1[i] == cls),
                'obj_top5': bool(cls in obj_top5[i]),
                'bg_top1': bool(bg_top1[i] == cls),
                'bg_top5': bool(cls in bg_top5[i]),
                'conf_orig': orig_conf[i, cls].item(),
                'conf_obj': obj_conf[i, cls].item(),
                'conf_bg': bg_conf[i, cls].item(),
            }

        # Print summary
        orig_acc1 = sum(1 for v in cache.values() if v['orig_top1']) / N * 100
        orig_acc5 = sum(1 for v in cache.values() if v['orig_top5']) / N * 100
        obj_acc1 = sum(1 for v in cache.values() if v['obj_top1']) / N * 100
        bg_acc1 = sum(1 for v in cache.values() if v['bg_top1']) / N * 100
        print(f"  Orig: Top1={orig_acc1:.1f}%, Top5={orig_acc5:.1f}%")
        print(f"  Obj:  Top1={obj_acc1:.1f}%")
        print(f"  BG:   Top1={bg_acc1:.1f}%")

        # Free GPU memory
        del train_features, train_labels
        torch.cuda.empty_cache()

    print("\nAll kNN inference completed.")
    save_inference_cache(inference_cache, save_path=save_path)


if __name__ == '__main__':
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset_test = ImageNet("/home/dataset/imagenet", split="val", transform=val_transform)
    dataset_train = ImageNet("/home/dataset/imagenet", split="train", transform=train_transform)
    bbox_dir = "/home/dataset/imagenet/ILSVRC/Annotations/CLS-LOC/val"

    # Train loader (deterministic: no random crop/flip)
    train_loader = DataLoader(dataset_train, batch_size=512, shuffle=False,
                              num_workers=8, pin_memory=True)

    # Val set with masking
    all_data_list = []
    print("Collecting images with bounding boxes for all classes...")
    for idx in tqdm(range(len(dataset_test))):
        image_path, label = dataset_test.samples[idx]
        img_name = os.path.basename(image_path)
        xml_name = os.path.splitext(img_name)[0] + ".xml"
        xml_path = os.path.join(bbox_dir, xml_name)
        if os.path.exists(xml_path):
            all_data_list.append({
                'image_path': image_path,
                'xml_path': xml_path,
                'class': label,
            })

    masking_dataset = MaskedImageDataset(all_data_list, val_transform)
    masking_loader = DataLoader(masking_dataset, batch_size=512, shuffle=False,
                                num_workers=8, pin_memory=True, persistent_workers=True)

    # ===== Configuration =====
    steps_int = [150, 200, 250, 300, 350, 400, 450, 500]
    steps = [str(i) for i in steps_int]
    k = 20  # number of neighbors
    T = 0.07  # temperature

    configs = [
        {
            "name": "lambda0.005",
            "model_path": "/models/jhkim/imagenet500_lambda0.005",
            "save_path": "knn_inference_lambda0.005_cache.pkl",
        },
        {
            "name": "lambda0.01",
            "model_path": "/models/jhkim/imagenet500_lambda0.01",
            "save_path": "knn_inference_lambda0.01_cache.pkl",
        },
        {
            "name": "lambda0.05",
            "model_path": "/models/jhkim/imagenet500_lambda0.05",
            "save_path": "knn_inference_lambda0.05_cache.pkl",
        },
    ]

    for cfg in configs:
        print(f"\n{'=' * 60}")
        print(f"  Loading models for: {cfg['name']} ({cfg['model_path']})")
        print(f"{'=' * 60}")

        models_list = []
        for step in steps:
            resnet = torchvision.models.resnet50(weights=None)
            resnet.fc = nn.Identity()
            load_model(resnet, f"{cfg['model_path']}/{step}.pth")
            models_list.append(resnet)

        knn_inference(
            steps=steps,
            models_steps=models_list,
            save_path=cfg["save_path"],
            masking_dataset=masking_dataset,
            masking_loader=masking_loader,
            train_loader=train_loader,
            k=k,
            T=T,
        )