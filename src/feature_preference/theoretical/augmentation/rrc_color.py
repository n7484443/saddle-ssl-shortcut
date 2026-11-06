
from pathlib import Path
import sys

src_dir = next(p for p in Path(__file__).resolve().parents
               if (p / "common.py").is_file())
sys.path.insert(0, str(src_dir))

import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset
import random
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from setproctitle import setproctitle
from tqdm import tqdm
import os
import random

import math
import torchvision
from common import *  # from your_loss_file import BTLoss

# --- ResNet9 Architecture ---
class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, x): return x * self.weight


class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)


class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x): return x + self.module(x)


def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, pools=0):
    layers = []
    layers.append(nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding, bias=False))
    layers.append(nn.BatchNorm2d(channels_out))
    layers.append(nn.ReLU(inplace=True))
    if pools > 0: layers.append(nn.MaxPool2d(pools))
    return nn.Sequential(*layers)

class SyntheticColorDataset(Dataset):
    def __init__(self, size_dataset, colors_center, colors_outer, configs, split='train', transform=None, no_aug=False):
        self.size_dataset = size_dataset
        self.split = split
        self.no_aug = no_aug

        # Dimensions
        self.m = 96
        self.a = 16
        self.b = 16

        self.colors_center = colors_center
        self.colors_outer = colors_outer

        # Pre-generate indices: (background_idx, center_idx, outer_idx)
        self.data_info = []
        self.datas = []
        mean = 0.0
        var = 0.0
        for _ in range(size_dataset):
            center_idx = random.randint(0, len(self.colors_center)-1)
            outer_idx = random.randint(0, len(self.colors_outer)-1)
            self.data_info.append((center_idx, outer_idx))
            imgs = self._generate_image(center_idx, outer_idx)
            self.datas.append(imgs)
            image_preprocess = transforms.ToTensor()(imgs)
            mean += image_preprocess.mean(dim=[1, 2])
            var += image_preprocess.var(dim=[1, 2])

        mean /= len(self.datas)
        var /= len(self.datas)
        std = torch.sqrt(var)
        print(mean, std)

        # Transforms
        # mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        min_scale = float(configs.crop_min)
        print(min_scale)
        self.augment = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomApply([transforms.GaussianNoise(sigma=0.05)], p=0.5),
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(self.m, scale=(min_scale, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.augment_prime = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomApply([transforms.GaussianNoise(sigma=0.05)], p=0.5),
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(self.m, scale=(min_scale, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def _generate_image(self, center_idx, outer_idx):
        # Background
        img = torch.zeros(3, self.m, self.m)

        radius = round(math.sqrt(self.a * self.a / math.pi))

        y = torch.arange(self.m)
        x = torch.arange(self.m)
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        jitter_range = 8
        offset_y = random.randint(-jitter_range, jitter_range)
        offset_x = random.randint(-jitter_range, jitter_range)

        cy = (self.m // 2) + offset_y
        cx = (self.m // 2) + offset_x

        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        mask = dist_sq <= radius ** 2

        for c in range(3):
            img[c, mask] = self.colors_center[center_idx][c]

        # # Center Square (ax8)
        # start_a = (self.m - self.a) // 2
        # img[:, start_a:start_a + self.a, start_a:start_a + self.a] = self.colors_center[center_idx].view(3, 1, 1)

        # Outer Square (bxb) - positioned at top-left with slight margin
        margin_x = 10
        margin_y = 10
        corners = [
            (margin_x, margin_y),  # Top-Left
            (margin_x, self.m - self.b - margin_y),  # Bottom-Left
            (self.m - self.b - margin_x, margin_y),  # Top-Right
            (self.m - self.b - margin_x, self.m - self.b - margin_y)  # Bottom-Right
        ]
        start_y, start_x = random.choice(corners)
        img[:, start_y:start_y + self.b, start_x:start_x + self.b] = self.colors_outer[outer_idx].view(3, 1, 1)
        noise = torch.randn_like(img) * 0.05
        img = torch.clamp(img + noise, 0, 1)

        return transforms.ToPILImage()(img)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        center_idx, outer_idx = self.data_info[idx]
        image_pil = self.datas[idx]

        # Label1: center color (main task), Label2: outer color (shortcut)

        if self.no_aug:
            return self.transform(image_pil), center_idx, outer_idx
        if self.split == 'train':
            return self.augment(image_pil), self.augment_prime(image_pil)
        else:
            return self.transform(image_pil), center_idx, outer_idx

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 16, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 32x32

            # Layer 2
            nn.Conv2d(16, 32, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 16x16

            # Layer 3
            nn.Conv2d(32, 64, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 32x32

            # # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> 1x1x64
            nn.Flatten()
        )
        self.size_in = 128
        # self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.backbone.maxpool = nn.Identity()

        # self.backbone = torchvision.models.resnet18(zero_init_residual=True, weights=None)
        # self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.backbone.maxpool = nn.Identity()
        # self.size_in = self.backbone.fc.in_features
        self.size_out = 32
        self.backbone.fc = nn.Identity()

        # Projector
        sizes = [self.size_in, self.size_out]
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
        return self.projector(self.backbone(x))


def get_synthetic_dataloaders(configs, num_classes):
    bs = int(configs.batch_size)

    # Define sizes
    train_len = 5000
    test_len = 1000

    colors_center = torch.rand(num_classes, 3)
    colors_outer = torch.rand(num_classes, 3)

    train_ds = SyntheticColorDataset(train_len, colors_center, colors_outer, configs, split='train', no_aug=not configs.augmentation)
    test_ds = SyntheticColorDataset(test_len, colors_center, colors_outer, configs, split='test', no_aug=True)
    train_noaug_ds = SyntheticColorDataset(train_len, colors_center, colors_outer, configs, split='train', no_aug=True)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    train_noaug_loader = DataLoader(train_noaug_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)

    return train_loader, test_loader, train_noaug_loader


def train_eval(configs, train, test, train_noaug, epochs, file_path, save_path, num_classes, lambd=None):
    model = Model().to(device)
    size_in = model.size_in
    scale_weights(model, configs.scaling)

    base_lr = float(configs.lr)
    loss_fn = BTLoss(use_batchnorm=True, in_feature=model.size_out, lambd=float(lambd)).to(device)

    optimizer_main = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-6)

    linear_center = nn.Linear(size_in, num_classes).to(device)
    linear_outer = nn.Linear(size_in, num_classes).to(device)

    state_dict_center = linear_center.state_dict()
    state_dict_outer = linear_outer.state_dict()

    scheduler = torch.optim.lr_scheduler.ExponentialLR(gamma=configs.learning_rate_decay, optimizer=optimizer_main)

    bar = tqdm(range(epochs), mininterval=10, maxinterval=10)
    report_freq = int(configs.report_frequency)
    save_freq = int(configs.save_frequency)
    step = 1

    def update_accuracy_lin(current_steps):
        model.eval()
        # Reset probes
        linear_center.load_state_dict(state_dict_center)
        linear_outer.load_state_dict(state_dict_outer)

        opt_center = torch.optim.Adam(linear_center.parameters(), lr=5e-3)
        opt_outer = torch.optim.Adam(linear_outer.parameters(), lr=5e-3)

        linear_center.train()
        linear_outer.train()

        probe_epochs = 30
        final_loss_c = 0.0
        final_loss_o = 0.0
        for _ in tqdm(range(probe_epochs), mininterval=20, maxinterval=20):
            for img, center, outer in train_noaug:
                img, center, outer = img.to(device), center.to(device), outer.to(device)

                with torch.no_grad():
                    feats = model(img, use_projector=False)
                    feats = F.normalize(feats, dim=1)

                # Center Loss
                out_c = linear_center(feats)
                loss_c = F.cross_entropy(out_c, center)
                opt_center.zero_grad()
                loss_c.backward()
                opt_center.step()

                # Background Loss
                out_o = linear_outer(feats)
                loss_o = F.cross_entropy(out_o, outer)
                opt_outer.zero_grad()
                loss_o.backward()
                opt_outer.step()

                final_loss_c = loss_c.item()
                final_loss_o = loss_o.item()

        # Evaluation
        linear_center.eval()
        linear_outer.eval()
        acc_c, acc_o, total = 0.0, 0.0, 0.0

        with torch.no_grad():
            for img, center, outer in test:
                img, center, outer = img.to(device), center.to(device), outer.to(device)
                feats = model(img, use_projector=False)
                feats = F.normalize(feats, dim=1)

                pred_c = torch.argmax(linear_center(feats), dim=1)
                pred_o = torch.argmax(linear_outer(feats), dim=1)

                acc_c += (pred_c == center).float().sum().item()
                acc_o += (pred_o == outer).float().sum().item()
                total += center.shape[0]

        acc_c = acc_c / total * 100.0
        acc_o = acc_o / total * 100.0
        write_realtime_seperate(file_path, "accuracy", [acc_c, acc_o, current_steps, final_loss_c, final_loss_o])
        model.train()
        return acc_c

    # Initial check
    # update_accuracy_lin(0)
    for epoch in bar:
        losses = []
        for (img1, img2) in train:
            img1, img2 = img1.to(device), img2.to(device)

            z1, z2 = model(torch.cat([img1, img2], dim=0)).chunk(2, dim=0)
            c, loss = loss_fn(z1, z2)

            optimizer_main.zero_grad()
            loss.backward()
            optimizer_main.step()
            losses.append(loss.item())
            if step % save_freq == 0:
                os.makedirs(save_path, exist_ok=True)
                torch.save(model.state_dict(), f"{save_path}/{step}.pt")

            if step % report_freq == 0:
                with torch.no_grad():
                    eig = (torch.linalg.svdvals(c) ** 2).cpu()
                    write_realtime_seperate(file_path, "eig", eig)
                acc = update_accuracy_lin(step)
                bar.set_description(f"Loss: {np.mean(losses):.4f} | Acc: {acc:.2f}")
            step += 1

        bar.set_description(f"Loss: {np.mean(losses):.4f}")


        scheduler.step()
        write_realtime_list(file_path, "loss", losses)


@hydra.main(version_base=None, config_path="configs")
def run(configs):
    global device
    device = torch.device("cuda")
    seed = int(configs.seed)
    set_seed(seed)

    lambd = configs.lambdas
    epochs = configs.epochs
    num_classes = 10

    setproctitle(f"BT_Res18_{epochs}")
    train, test, train_noaug = get_synthetic_dataloaders(configs, num_classes=num_classes)
    set_seed(seed)

    save_dir = f"datas/{configs.method}_{lambd}/{seed}"
    model_dir = f"/models/jhkim/rrc/models/{configs.method}_{lambd}/{seed}"

    train_eval(configs, train, test, train_noaug, epochs, save_dir, model_dir, lambd=lambd, num_classes=num_classes)


if __name__ == "__main__":
    run()