
from pathlib import Path
import sys

src_dir = next(p for p in Path(__file__).resolve().parents
               if (p / "common.py").is_file())
sys.path.insert(0, str(src_dir))

import torchvision.datasets
from PIL.Image import Image
from setproctitle import setproctitle
import torch.utils.data
import torchvision.transforms.v2 as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image, ImageOps, ImageFilter
import hydra
from omegaconf import DictConfig
import copy

from common import *

device = torch.device("cuda")

class CMNISTDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None, no_aug=False, mean=None, std=None):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.no_aug = no_aug
        if train:
            self.dataset_size = [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]
        else:
            self.dataset_size = [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]
        self.dataset_type = dataset_type

        to_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.labels = []
        self.img = []

        for t in range(self.dataset_type):
            for item in range(self.dataset_size[t]):
                self.labels.append(t)
                image = Image.open(
                    os.path.join(self.root_dir, f'{'train' if train else 'test'}/{t}/{item}.png')).convert('RGB')
                self.img.append(image)

        if mean is not None and std is not None:
            self.mean = mean
            self.std = std
        else:
            # Pass 1: mean
            pixel_sum = torch.zeros(3)
            pixel_count = 0
            for image in self.img:
                tensor = to_tensor(image)  # [3, H, W]
                pixel_sum += tensor.sum(dim=[1, 2])
                pixel_count += tensor.shape[1] * tensor.shape[2]
            self.mean = pixel_sum / pixel_count

            # Pass 2: variance
            pixel_var_sum = torch.zeros(3)
            for image in self.img:
                tensor = to_tensor(image)
                diff = tensor - self.mean.view(3, 1, 1)
                pixel_var_sum += (diff ** 2).sum(dim=[1, 2])
            self.std = torch.sqrt(pixel_var_sum / pixel_count)

        print(f"mean: {self.mean}, std: {self.std}")

        noise_var = 0.1
        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
            transforms.ToTensor(),
            transforms.RandomApply(
                [transforms.GaussianNoise(sigma=noise_var)],
                p=0.5
            ),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        self.augment_prime = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
            transforms.ToTensor(),
            transforms.RandomApply(
                [transforms.GaussianNoise(sigma=noise_var)],
                p=0.5
            ),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        image_original = self.img[idx]
        if self.no_aug:
            return self.transform(image_original), self.labels[idx]
        else:
            return self.augment(image_original), self.augment_prime(image_original)

def get_cmnist_dataloaders(root_dir, train_batch_size, test_batch_size):
    train_dataset = CMNISTDataset(
        root_dir=root_dir,
        train=True,
    )

    train_mean = train_dataset.mean
    train_std = train_dataset.std

    test_dataset = CMNISTDataset(
        root_dir=root_dir,
        train=False,
        no_aug=True,
        mean=train_mean,
        std=train_std,
    )

    train_noaug_dataset = CMNISTDataset(
        root_dir=root_dir,
        train=True,
        no_aug=True,
        mean=train_mean,
        std=train_std,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
    )

    train_noaug_loader = DataLoader(
        train_noaug_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    return train_loader, test_loader, train_noaug_loader


class BarlowTwins(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.network == "ResNet-18":
            self.backbone = torchvision.models.resnet18(zero_init_residual=cfg.zero_init_residual, weights=None)
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.backbone.maxpool = nn.Identity()
            self.size_in = self.backbone.fc.in_features
            # self.size_out = 128
            self.size_out = 1024
            self.backbone.fc = nn.Identity()
        else:
            self.backbone = torchvision.models.vgg11_bn(weights=None)
            self.size_in = 512
            self.size_out = 128
            self.backbone.classifier = nn.Identity()
            self.backbone.features[3] = nn.Identity()
            self.backbone.features[20] = nn.Identity() #remove last maxpool
            self.backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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


def import_dataset(train_batch_size, test_batch_size, ratio):
    root_dir = f"/home/dataset/cmnist/{ratio}"

    train_loader, test_loader, train_noaug_loader = get_cmnist_dataloaders(
        root_dir=root_dir,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
    )
    print(f"train dataset size: {len(train_loader.dataset)}")
    print(f"test dataset size: {len(test_loader.dataset)}")
    return train_loader, test_loader, train_noaug_loader

def train_eval(cfg, train, test, train_noaug, epochs, file_path, lambd=None):
    model = BarlowTwins(cfg).to(device)
    size_in = model.size_in
    size_out = model.size_out

    scale_weights(model, float(cfg.scaling), scale_bias=True)
    base_lr = float(cfg.lr)
    if method == "BarlowTwins":
        loss_fn = BTLoss(use_batchnorm=True, in_feature=size_out, lambd=float(lambd)).to(device)
        # loss_fn = BTLoss(use_batchnorm=False, in_feature=size_out, lambd=float(lambd)).to(device)
    elif method == "VICReg":
        loss_fn = VICRegLoss().to(device)
    elif method == "LeJEPA":
        loss_fn = LeJEPALoss().to(device)
    else:
        loss_fn = SimCLRLoss(batch_size, temperature=float(cfg.temperature)).to(device)
    if cfg.optimizer == "SGD":
        optimizer_main = torch.optim.SGD(model.parameters(), lr=base_lr, weight_decay=weight_decay, momentum=0.9)
        # optimizer_main = torch.optim.SGD(model.parameters(), lr=base_lr, weight_decay=weight_decay, momentum=0)
    else:
        optimizer_main = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    linear_regression = nn.Linear(size_in, dataset_type).to(device)
    state_dict = copy.deepcopy(linear_regression.state_dict())

    bar = tqdm(range(epochs), mininterval=10, maxinterval=10)

    # =========================================================================
    # MODIFIED: Feature-cached linear probing
    # Instead of running backbone forward 30 times (once per LP epoch),
    # we extract features once, cache them, then train the linear head
    # on the cached features. This cuts backbone forward cost by ~30x.
    # =========================================================================
    def update_accuracy_lin(current_steps):
        model.eval()

        # --- Phase 1: Cache ALL backbone features on GPU (no CPU round-trip) ---
        all_features_train = []
        all_labels_train = []
        all_features_test = []
        all_labels_test = []

        with torch.no_grad():
            for (image, label) in train_noaug:
                image = image.to(device, non_blocking=True)
                all_features_train.append(model(image, use_projector=False))
                all_labels_train.append(label.to(device, non_blocking=True))
            for (image, label) in test:
                image = image.to(device, non_blocking=True)
                all_features_test.append(model(image, use_projector=False))
                all_labels_test.append(label.to(device, non_blocking=True))

        cached_features_train = torch.cat(all_features_train, dim=0)
        cached_labels_train = torch.cat(all_labels_train, dim=0)
        cached_features_test = torch.cat(all_features_test, dim=0)
        cached_labels_test = torch.cat(all_labels_test, dim=0)

        # Free intermediate lists
        del all_features_train, all_labels_train, all_features_test, all_labels_test

        # --- Phase 2: Train linear classifier on cached GPU features ---
        n_train = cached_features_train.shape[0]
        lp_batch_size = train_noaug.batch_size

        linear_regression.load_state_dict(state_dict)
        optimizer_classifier = torch.optim.Adam(linear_regression.parameters(), lr=1e-3)
        linear_regression.train()

        lp_bar = tqdm(range(35), mininterval=10, maxinterval=10)
        g = torch.Generator(device=device)
        g.manual_seed(0)
        fixed_perm = torch.randperm(n_train, generator=g, device=device)
        for _ in lp_bar:
            # Shuffle indices on GPU — no DataLoader overhead
            total = 0.0
            sum_loss = 0.0
            for i in range(0, n_train, lp_batch_size):
                idx = fixed_perm[i:i + lp_batch_size]
                feat = cached_features_train[idx]
                label = cached_labels_train[idx]
                outputs = linear_regression(feat)
                loss = F.cross_entropy(outputs, label)
                optimizer_classifier.zero_grad()
                loss.backward()
                optimizer_classifier.step()
                total += feat.shape[0]
                sum_loss += feat.shape[0] * loss.item()
            lp_bar.set_description(f"loss linear : {sum_loss / total}", refresh=False)

        # --- Phase 3: Evaluate on cached test features (single forward) ---
        linear_regression.eval()
        with torch.no_grad():
            pred = linear_regression(cached_features_test).argmax(dim=1)
            acc_test = (pred == cached_labels_test).float().mean().item() * 100.0

        # Free cached tensors
        del cached_features_train, cached_labels_train
        del cached_features_test, cached_labels_test

        write_realtime(file_path, "accuracy", acc_test)
        write_realtime(file_path, "steps", current_steps)
        model.train()
        return acc_test

    update_accuracy_lin(current_steps=0)
    step = 1
    model.train()
    for _ in bar:
        total_size = 0.0
        loss_total = 0.0
        losses = []
        for (image1, image2) in train:
            image1 = image1.to(device, non_blocking=True)
            image2 = image2.to(device, non_blocking=True)
            outputs = model(torch.cat([image1, image2], dim=0))
            z1, z2 = outputs.chunk(2, dim=0)
            _, loss = loss_fn(z1, z2)
            optimizer_main.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer_main.step()

            total_size += image1.shape[0]
            loss_total += loss.item()
            if step % cfg.report_frequency == 0:
                update_accuracy_lin(current_steps=step)
            step += 1
        write_realtime_list(file_path, "loss", losses)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    global method, network, dataset_type, input_size, weight_decay, batch_size

    method = cfg.method
    network = cfg.network
    dataset_type = 10
    input_size = cfg.input_size

    ratio = cfg.ratio
    lambdas = cfg.lambdas
    epochs = cfg.epochs

    weight_decay = float(cfg.weight_decay)
    batch_size = int(cfg.batch_size)

    setproctitle(f"cmnist-{ratio}")
    print(f"lambda: {lambdas}, ratio: {ratio}")
    seed = int(cfg.seed)
    set_seed(seed)
    train, test, train_noaug = import_dataset(train_batch_size=batch_size, test_batch_size=512, ratio=ratio)
    set_seed(seed)
    train_eval(cfg, train, test, train_noaug, epochs, f"datas/{method}/{seed}/ratio_{ratio}_lambda_{lambdas}_lr_{cfg.lr:.1e}", lambd=lambdas)


if __name__ == "__main__":
    main()