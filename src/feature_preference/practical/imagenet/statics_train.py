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


class LinearClassifier(nn.Module):
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


def load_ddp_checkpoint(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    return model


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


def denormalize(tensor):
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1).to(tensor.device)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1).to(tensor.device)
    denorm = tensor * std + mean
    return torch.clamp(denorm, 0, 1)


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

        # Apply transform to image
        img_tensor = self.transform(img)
        H, W = img_tensor.shape[1], img_tensor.shape[2]  # 224, 224

        # Create bbox mask on PIL Image, then transform it
        bbox_mask_pil = Image.new('L', (orig_w, orig_h), 0)
        draw = ImageDraw.Draw(bbox_mask_pil)

        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox
            draw.rectangle([xmin, ymin, xmax, ymax], fill=255)

        # Apply same transform to mask (without normalization)
        mask_transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        bbox_mask_tensor = mask_transform(bbox_mask_pil)  # (1, 224, 224)
        bbox_mask_tensor = (bbox_mask_tensor > 0.5).float()  # Binarize

        # Create 3-channel mask
        mask = 1 - bbox_mask_tensor.repeat(3, 1, 1)  # 0 where bbox, 1 elsewhere

        object_tensor = img_tensor * (1 - mask)
        background_tensor = img_tensor * mask

        return img_tensor, object_tensor, background_tensor, data['class'], mask, idx


def inference(steps, models_steps, linear_steps, save_path, masking_loader, masking_dataset):
    inference_cache = {step: {} for step in steps}

    for step, model, linear in zip(steps, models_steps, linear_steps):
        print(f"\n[Step {step}] Running Inference & Caching Results...")
        model.to(device).eval()
        linear.to(device).eval()
        model_grad = torch.nn.Sequential(model, linear)

        list_classes = []
        list_conf = []
        list_top1 = []
        list_top5 = []

        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            for orig_batch, object_batch, background_batch, classes, _, indices in tqdm(masking_loader,
                                                                                        desc=f"Inferencing {step}"):
                B = orig_batch.size(0)

                combined = torch.cat([orig_batch, object_batch, background_batch], dim=0).to(device, non_blocking=True)
                classes = classes.to(device, non_blocking=True)

                logits_all = model_grad(combined)

                classes_3x = classes.repeat(3)
                idx_3x = torch.arange(3 * B, device=device)

                log_probs = logits_all - logits_all.logsumexp(dim=1, keepdim=True)
                conf_all = log_probs[idx_3x, classes_3x].exp().view(3, B)

                preds = logits_all.argmax(dim=1).view(3, B)
                corr_top1 = (preds == classes)

                top5 = logits_all.topk(5, dim=1, sorted=False)[1].view(3, B, 5)
                corr_top5 = (top5 == classes.view(1, B, 1)).any(dim=2)

                list_classes.append(classes)
                list_conf.append(conf_all)
                list_top1.append(corr_top1)
                list_top5.append(corr_top5)

        all_classes_np = torch.cat(list_classes).cpu().numpy()
        all_conf_np = torch.cat(list_conf, dim=1).cpu().numpy()
        all_top1_np = torch.cat(list_top1, dim=1).cpu().numpy()
        all_top5_np = torch.cat(list_top5, dim=1).cpu().numpy()

        cache = inference_cache[step]
        for i in range(len(masking_dataset)):
            cache[i] = {
                'class': all_classes_np[i],
                'orig_top1': all_top1_np[0, i], 'orig_top5': all_top5_np[0, i],
                'obj_top1': all_top1_np[1, i], 'obj_top5': all_top5_np[1, i],
                'bg_top1': all_top1_np[2, i], 'bg_top5': all_top5_np[2, i],
                'conf_orig': all_conf_np[0, i],
                'conf_obj': all_conf_np[1, i],
                'conf_bg': all_conf_np[2, i]
            }

    print("All inference completed and cached.")

    save_inference_cache(inference_cache, save_path=save_path)


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


def load_models_for_config(model_path, steps):
    """Load backbone and linear classifier for each step in a given config."""
    models_list = []
    linears_list = []
    for step in steps:
        resnet = torchvision.models.resnet50(weights=None)
        embed_dim = resnet.fc.weight.shape[1]
        resnet.fc = nn.Identity()
        load_model(resnet, f"{model_path}/{step}.pth")
        models_list.append(resnet)

        linear = LinearClassifier(embed_dim)
        load_ddp_checkpoint(linear, f"{model_path}/{step}/final_checkpoint.pth")
        linears_list.append(linear)
    return models_list, linears_list


if __name__ == '__main__':
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset_test = ImageNet("/home/dataset/imagenet", split="val", transform=val_transform)
    bbox_dir = "/home/dataset/imagenet/ILSVRC/Annotations/CLS-LOC/val"

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
    masking_loader = DataLoader(masking_dataset, batch_size=512, shuffle=False, num_workers=8, pin_memory=True,
                                persistent_workers=True)

    # ===== Configuration: define all experiments here =====
    steps_int = [5 * i for i in range(0, 11)]
    steps = [str(i) for i in steps_int]

    configs = [
        {
            "name": "lambda0.005",
            "model_path": "/models/jhkim/imagenet200_lambda0.005",
            "save_path": "inference_lambda0.005_cache.pkl",
        },
        {
            "name": "lambda0.01",
            "model_path": "/models/jhkim/imagenet200_lambda0.01",
            "save_path": "inference_lambda0.01_cache.pkl",
        },
        {
            "name": "lambda0.05",
            "model_path": "/models/jhkim/imagenet200_lambda0.05",
            "save_path": "inference_lambda0.05_cache.pkl",
        },
    ]

    # ===== Run inference for each config =====
    for cfg in configs:
        print(f"\n{'=' * 60}")
        print(f"  Loading models for: {cfg['name']} ({cfg['model_path']})")
        print(f"{'=' * 60}")

        models_list, linears_list = load_models_for_config(cfg["model_path"], steps)

        inference(
            steps=steps,
            models_steps=models_list,
            linear_steps=linears_list,
            save_path=cfg["save_path"],
            masking_loader=masking_loader,
            masking_dataset=masking_dataset,
        )
