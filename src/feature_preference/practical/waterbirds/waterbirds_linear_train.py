
from pathlib import Path
import sys

src_dir = next(p for p in Path(__file__).resolve().parents
               if (p / "common.py").is_file())
sys.path.insert(0, str(src_dir))

from tqdm import tqdm
import torch.nn.functional as F
from common import *
import os
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from PIL.Image import Image
from PIL import Image
import pandas as pd
import torchvision

ratios = [0.05, 0.075, 0.1, 0.2]
lambdas = 0.05
loss = []
acc_object = []
acc_background = []
steps = []
eig_list = []
method = "BarlowTwins_single_background"
directory_models = f"/models/jhkim/waterbirds/{method}_{lambdas}"
directory = f"datas/{method}_{lambdas}"

device = torch.device("cuda")
size = 128


class WaterbirdDataset(Dataset):
    def __init__(self, root_dir, metadata_path, split='train', transform=None, balance_classes=True, no_aug=False):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.no_aug = no_aug
        self.metadata = pd.read_csv(metadata_path)

        split_dict = {'train': 0, 'val': 1, 'test': 2}

        self.metadata = self.metadata[self.metadata['split'] == split_dict[split]]
        if balance_classes:
            landbird_data = self.metadata[self.metadata['y'] == 0]
            waterbird_data = self.metadata[self.metadata['y'] == 1]

            min_class_size = min(len(landbird_data), len(waterbird_data))
            # min_class_size = 256

            landbird_balanced = landbird_data.sample(min_class_size, random_state=1)
            waterbird_balanced = waterbird_data.sample(min_class_size, random_state=1)

            self.metadata = pd.concat([landbird_balanced, waterbird_balanced])

        self.metadata = self.metadata.reset_index(drop=True)

        self.size_transform = transforms.Resize(size)

        self.labels = []
        self.places = []
        self.img_ids = []
        self.img = []
        mean = 0.0
        var = 0.0

        for idx in range(len(self.metadata)):
            img_filename = self.metadata.loc[idx, 'img_filename']
            self.labels.append(int(self.metadata.loc[idx, 'y']))
            self.places.append(int(self.metadata.loc[idx, 'place']))
            self.img_ids.append(self.metadata.loc[idx, 'img_id'] if 'img_id' in self.metadata.columns else idx)
            image = Image.open(os.path.join(self.root_dir, img_filename)).convert('RGB')
            image_resize = self.size_transform(image)
            self.img.append(image_resize)
            image_preprocess = transforms.ToTensor()(image_resize)
            mean += image_preprocess.mean(dim=[1, 2])
            var += image_preprocess.var(dim=[1, 2])

        mean /= len(self.img)
        var /= len(self.img)
        std = torch.sqrt(var)
        print(mean, std)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.transform_image = transforms.ToTensor()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_original = self.img[idx]
        label = self.labels[idx]
        places = self.places[idx]
        if self.split == 'test':
            return self.transform(image_original), self.transform_image(image_original), label, places
        return self.transform(image_original), label, places

class BarlowTwins(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet34(zero_init_residual=True, weights=None)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.size_in = self.backbone.fc.in_features
        self.size_out = 256
        self.backbone.fc = nn.Identity()

        sizes = [self.size_in, self.size_out, self.size_out]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

    def forward(self, x, use_projector=True):
        if not use_projector:
            return self.backbone(x)
        z = self.projector(self.backbone(x))
        return z

def generate_resnet_from_dict(path):
    model = BarlowTwins()
    model.load_state_dict(torch.load(path))
    return model.backbone, model.size_in

image_map = {}
pt_list = range(40, 40+40*20+1, 40)
for ratio in ratios:
    root_dir = f"/home/dataset/waterbird_single_background/{ratio}"
    metadata_path = f"/home/dataset/waterbird_single_background/{ratio}/metadata.csv"
    train_noaug_dataset = WaterbirdDataset(
        root_dir=root_dir,
        metadata_path=metadata_path,
        split='train',
        balance_classes=True,
    )
    train_noaug_loader = DataLoader(
        train_noaug_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=6
    )

    image_map[ratio] = []
    for i in range(len(pt_list)):
        print(f"Processing {ratios}: {pt_list[i]}")
        model, size_out = generate_resnet_from_dict(f"{directory_models}/{ratio}/{pt_list[i]}.pt")
        model = model.to(device)
        linear = nn.Linear(size_out, 2).to(device)

        optimizer_linear = torch.optim.Adam(linear.parameters())
        linear.train()
        bar = tqdm(range(32), mininterval=10, maxinterval=10)
        for epoch in bar:
            loss_total = 0.0
            for image, label, _ in train_noaug_loader:
                with torch.no_grad():
                    image = image.to(device)
                    label = label.to(device)
                    out = model(image)
                out = linear(out)
                loss_linear = F.cross_entropy(out, label)
                optimizer_linear.zero_grad()
                loss_linear.backward()
                loss_total += loss_linear.item()
                optimizer_linear.step()
            bar.set_description(f"loss : {loss_total / len(train_noaug_loader)}", refresh=False)
        save_path = f"{directory_models}/{ratio}/linear"
        os.makedirs(save_path, exist_ok=True)
        torch.save(linear.state_dict(), f"{save_path}/{pt_list[i]}.pt")