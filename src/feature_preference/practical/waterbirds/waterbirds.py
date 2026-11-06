
from pathlib import Path
import sys

src_dir = next(p for p in Path(__file__).resolve().parents
               if (p / "common.py").is_file())
sys.path.insert(0, str(src_dir))

import hydra
import torch
import torch.nn as nn
import torchvision.datasets
from PIL.Image import Image
from omegaconf import OmegaConf
from setproctitle import setproctitle
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms.v2 as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps, ImageFilter
import os
import pandas as pd
import random
import yaml

from common import *

class WaterbirdDataset(Dataset):
    def __init__(self, root_dir, metadata_path, size, configs, split='train', transform=None, balance_classes=True, no_aug=False):
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

            if configs.full_batch:
                min_class_size = configs.batch_size // 2
            else:
                min_class_size = min(len(landbird_data), len(waterbird_data))
            # min_class_size = 256

            landbird_balanced = landbird_data.sample(min_class_size, random_state=1)
            waterbird_balanced = waterbird_data.sample(min_class_size, random_state=1)

            self.metadata = pd.concat([landbird_balanced, waterbird_balanced])

        self.metadata = self.metadata.reset_index(drop=True)
        self._print_distribution(self.metadata)
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
        min_scale = float(configs.crop_min) if "crop_min" in configs else 0.7
        print(f"scale : {min_scale}")
        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(min_scale, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            # GaussianBlur(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.augment_prime = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(min_scale, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            # GaussianBlur(p=0.1),
            # Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def _print_distribution(self, metadata):
        landbird_land = len(metadata[(metadata['y'] == 0) & (metadata['place'] == 0)])
        landbird_water = len(metadata[(metadata['y'] == 0) & (metadata['place'] == 1)])
        waterbird_land = len(metadata[(metadata['y'] == 1) & (metadata['place'] == 0)])
        waterbird_water = len(metadata[(metadata['y'] == 1) & (metadata['place'] == 1)])

        total = len(metadata)

        print(f"Landbird + Land background:  {landbird_land:4d} ({landbird_land / total * 100:.1f}%)")
        print(f"Landbird + Water background: {landbird_water:4d} ({landbird_water / total * 100:.1f}%)")
        print(f"Waterbird + Land background: {waterbird_land:4d} ({waterbird_land / total * 100:.1f}%)")
        print(f"Waterbird + Water background:{waterbird_water:4d} ({waterbird_water / total * 100:.1f}%)")
        print(f"Total: {total}")

        y_counts = metadata['y'].value_counts().sort_index()
        place_counts = metadata['place'].value_counts().sort_index()
        print(f"Bird type - Landbird: {y_counts.get(0, 0)}, Waterbird: {y_counts.get(1, 0)}")
        print(f"Background - Land: {place_counts.get(0, 0)}, Water: {place_counts.get(1, 0)}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_original = self.img[idx]
        label = self.labels[idx]
        places = self.places[idx]

        if self.no_aug:
            return self.transform(image_original), label, places
        if self.split == 'train':
            return self.augment(image_original), self.augment_prime(image_original)
        else:
            return self.transform(image_original), label, places

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

def get_waterbird_dataloaders(root_dir, metadata_path, test_batch_size, balance_class, configs):
    train_batch_size = int(configs.batch_size)
    if configs.full_batch:
        test_batch_size = train_batch_size

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = WaterbirdDataset(
        root_dir=root_dir,
        metadata_path=metadata_path,
        split='train',
        no_aug=not configs.augmentation,
        balance_classes=balance_class,
        size=configs.input_size,
        configs=configs,
    )

    test_dataset = WaterbirdDataset(
        root_dir=root_dir,
        metadata_path=metadata_path,
        split='train' if configs.use_trainset_on_test else 'test',
        transform=eval_transform,
        balance_classes=balance_class,
        no_aug=True,
        size=configs.input_size,
        configs=configs,
    )

    train_noaug_dataset = WaterbirdDataset(
        root_dir=root_dir,
        metadata_path=metadata_path,
        split='train',
        transform=eval_transform,
        balance_classes=balance_class,
        no_aug=True,
        size=configs.input_size,
        configs=configs,
    )

    # Create the DataLoader.
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=8,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    train_noaug_loader = DataLoader(
        train_noaug_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    return train_loader, test_loader, train_noaug_loader


class BarlowTwins(nn.Module):
    def __init__(self):
        super().__init__()
        # self.backbone = torchvision.models.vit_b_16(weights=None, image_size=size)
        # self.size_in = 768
        # self.size_out = 512
        # self.backbone.heads = nn.Identity()
        self.backbone = torchvision.models.resnet34(zero_init_residual=True, weights=None)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.size_in = self.backbone.fc.in_features
        self.size_out = 256
        self.backbone.fc = nn.Identity()

        # projector
        # sizes = [self.size_in, self.size_out, self.size_out, self.size_out]
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

def import_dataset(configs, test_batch_size, ratio=0.2, balance_class=False):
    root_dir = f"/home/dataset/{configs.dataset_directory}/{ratio}"
    metadata_path = f"/home/dataset/{configs.dataset_directory}/{ratio}/metadata.csv"

    train_loader, test_loader, train_noaug_loader = get_waterbird_dataloaders(
        root_dir=root_dir,
        metadata_path=metadata_path,
        test_batch_size=test_batch_size,
        balance_class=balance_class,
        configs=configs
    )
    print(f"train dataset: {len(train_loader.dataset)}")
    print(f"test dataset: {len(test_loader.dataset)}")
    return train_loader, test_loader, train_noaug_loader

def train_eval(configs, train, test, train_noaug, epochs, file_path, save_path, lambd=None):
    model = BarlowTwins().to(device)
    size_in = model.size_in
    size_out = model.size_out

    scale_weights(model, float(configs.scaling))
    base_lr = float(configs.lr)
    if "BarlowTwins" in configs.method:
        loss_fn = BTLoss(use_batchnorm=True, in_feature=size_out, lambd=float(lambd)).to(device)
    elif configs.method == "VICReg":
        loss_fn = VICRegLoss().to(device)
    else:
        loss_fn = SimCLRLoss(batch_size, temperature=float(configs.temperature)).to(device)
    optimizer_main = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-6)
    dataset_type = 2
    linear_regression_bird = nn.Linear(size_in, dataset_type).to(device)
    linear_regression_background = nn.Linear(size_in, dataset_type).to(device)
    state_dict_bird = linear_regression_bird.state_dict()
    state_dict_background = linear_regression_background.state_dict()
    learning_rate_schedular = torch.optim.lr_scheduler.ExponentialLR(gamma=configs.learning_rate_decay, optimizer=optimizer_main)

    bar = tqdm(range(epochs), mininterval=10, maxinterval=10)

    report_frequency = int(configs.report_frequency)
    save_frequency = int(configs.save_frequency)

    def update_accuracy_lin(current_steps):
        model.eval()
        linear_regression_bird.load_state_dict(state_dict_bird)
        linear_regression_background.load_state_dict(state_dict_background)
        optimizer_classifier_bird = torch.optim.Adam(linear_regression_bird.parameters())
        optimizer_classifier_background = torch.optim.Adam(linear_regression_background.parameters())
        linear_regression_bird.train()
        linear_regression_background.train()
        bar = tqdm(range(50), mininterval=10, maxinterval=10)
        for _ in bar:
            total = 0.0
            sum_bird = 0.0
            sum_background = 0.0
            for (image, label, places) in train_noaug:
                with torch.no_grad():
                    image = image.to(device, non_blocking=True)
                    labels = label.to(device, non_blocking=True)
                    places = places.to(device, non_blocking=True)
                    outputs = model(image, use_projector=False)
                    outputs = F.normalize(outputs, dim=1)
                outputs_bird = linear_regression_bird(outputs)
                outputs_background = linear_regression_background(outputs)

                loss_bird = F.cross_entropy(outputs_bird, labels)
                optimizer_classifier_bird.zero_grad()
                loss_bird.backward()
                optimizer_classifier_bird.step()

                loss_background = F.cross_entropy(outputs_background, places)
                optimizer_classifier_background.zero_grad()
                loss_background.backward()
                optimizer_classifier_background.step()
                total += 1
                sum_bird += loss_bird.item()
                sum_background += loss_background.item()
            bar.set_description(f"loss linear : {sum_bird / total} | {sum_background / total}", refresh=False)

        linear_regression_bird.eval()
        linear_regression_background.eval()
        acc_object = 0.0
        acc_background = 0.0
        total_test = 0.0
        with torch.no_grad():
            for (image, label, place) in tqdm(test):
                image = image.to(device, non_blocking=True)
                labels = label.to(device, non_blocking=True)
                places = place.to(device, non_blocking=True)

                outputs = model(image, use_projector=False)
                outputs = F.normalize(outputs, dim=1)

                outputs_bird = linear_regression_bird(outputs)
                outputs_background = linear_regression_background(outputs)

                pred_bird = torch.argmax(outputs_bird, dim=1)
                pred_background = torch.argmax(outputs_background, dim=1)

                object_correct = pred_bird == labels
                background_correct = pred_background == places

                acc_object += object_correct.float().sum().item()
                acc_background += background_correct.float().sum().item()
                total_test += labels.shape[0]
        acc_object = acc_object / total_test * 100.0
        acc_background = acc_background / total_test * 100.0
        write_realtime_seperate(file_path, "accuracy", [acc_object, acc_background, current_steps])
        model.train()
        return acc_object

    update_accuracy_lin(current_steps=0)
    step = 1
    model.train()
    for epoch in bar:
        total_size = 0.0
        loss_total = 0.0
        losses = []
        if not configs.augmentation:
            for (image, label, place) in train:
                image = image.to(device, non_blocking=True)
                outputs = model(image)
                c, loss = loss_fn(outputs, outputs)
                optimizer_main.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer_main.step()

                total_size += image.shape[0]
                loss_total += loss.item()
                if step % save_frequency == 0:
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(model.state_dict(), f"{save_path}/{step}.pt")
                if step % report_frequency == 0:
                    with torch.no_grad():
                        eigenvalues = (torch.linalg.svdvals(c) ** 2).cpu()
                        write_realtime_seperate(file_path, "eig", eigenvalues)
                    update_accuracy_lin(current_steps=step)
                step += 1
                learning_rate_schedular.step()
            write_realtime_list(file_path, "loss", losses)
        else:
            for (image1, image2) in train:
                image1 = image1.to(device, non_blocking=True)
                image2 = image2.to(device, non_blocking=True)
                outputs = model(torch.cat([image1, image2], dim=0))
                z1, z2 = outputs.chunk(2, dim=0)
                c, loss = loss_fn(z1, z2)
                optimizer_main.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer_main.step()

                total_size += image1.shape[0]
                loss_total += loss.item()
                if step % save_frequency == 0:
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(model.state_dict(), f"{save_path}/{step}.pt")
                if step % report_frequency == 0:
                    with torch.no_grad():
                        eigenvalues = (torch.linalg.svdvals(c) ** 2).cpu()
                        write_realtime_seperate(file_path, "eig", eigenvalues)
                    acc = update_accuracy_lin(current_steps=step)
                    if "break_point" in configs and acc > configs.break_point:
                        write_realtime_list(file_path, "loss", losses)
                        return
                step += 1
                learning_rate_schedular.step()
            write_realtime_list(file_path, "loss", losses)

@hydra.main(version_base=None, config_path="configs")
def run(configs):
    global device
    device = torch.device("cuda")

    ratio = configs.ratio
    seed = int(configs.seed)
    print(f"ratio: {ratio}, seed: {seed}")
    set_seed(seed)
    lambd = configs.lambdas
    epochs = configs.epochs

    setproctitle(f"{epochs}_{ratio}")
    train, test, train_noaug = import_dataset(configs, test_batch_size=512, ratio=ratio,
                                              balance_class=True)

    set_seed(seed)
    train_eval(configs, train, test, train_noaug, epochs, f"datas/{configs.method}_{lambd}/{seed}/{ratio}",
               f"/models/jhkim/waterbirds/{configs.method}_{lambd}/{seed}/{ratio}", lambd=lambd)


if __name__ == "__main__":
    run()