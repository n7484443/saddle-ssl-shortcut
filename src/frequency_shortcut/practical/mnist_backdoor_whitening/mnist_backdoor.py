from pathlib import Path
import sys

src_dir = next(p for p in Path(__file__).resolve().parents
               if (p / "common.py").is_file() and (p / "feature_preference").is_dir())
sys.path.insert(0, str(src_dir))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.v2 as transforms
import numpy as np
from tqdm import tqdm
import random
import os
import csv
import hydra
from omegaconf import DictConfig
import common

device = torch.device("cuda")

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


from scipy.linalg import hadamard


def get_hadamard_triggers(n_classes=10, intensity=1.0):
    H = hadamard(16)

    # Row 0 is all ones; use rows 1-10 instead.
    triggers = torch.tensor(H[1:n_classes + 1], dtype=torch.float32)
    triggers = triggers * intensity

    return triggers  # shape: (10, 16)


# ---------------------------------------------------------------- ZCA whitening

def compute_zca(images, eps=1e-2):
    """
    Compute ZCA whitening statistics from the training set.

    Args:
        images: (N, 1, 28, 28) tensor after trigger insertion and subset selection.
        eps: Regularization to limit amplification in low-variance directions.

    Returns:
        (mean, W), with mean shaped (784,) and symmetric W shaped (784, 784).
    """
    N = images.size(0)
    X = images.reshape(N, -1)

    mean = X.mean(dim=0)
    Xc = X - mean

    # Compute eigh in float64 for numerical stability.
    cov = (Xc.T @ Xc).double() / (N - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.clamp_min(0.0)

    # W = V diag(1/sqrt(lam + eps)) V^T
    W = (eigvecs * (1.0 / torch.sqrt(eigvals + eps))) @ eigvecs.T

    return mean.contiguous(), W.float().contiguous()


def apply_zca(images, mean, W):
    """Apply the ZCA matrix. W is symmetric, so no transpose is needed."""
    shape = images.shape
    X = images.reshape(shape[0], -1)
    return ((X - mean) @ W).reshape(shape)


import torchvision.transforms.functional as TF
from torchvision.transforms import RandomResizedCrop


class BackdoorMNIST(Dataset):
    def __init__(self, bd_path, cfg, train=True, mode='ssl', noise_std=0.1, subset_indices=None,
                 rrc_scale=(0.95, 1.0), rrc_ratio=(0.95, 1.05), zca_stats=None):
        mnist = torchvision.datasets.MNIST(root='/home/dataset/mnist', train=train, download=False)
        self.bd_classes = torch.load(bd_path, weights_only=True)
        self.intensity = cfg.intensity
        self.mode = mode
        self.noise_std = noise_std
        self.img_size = 28
        self.rrc_scale = rrc_scale
        self.rrc_ratio = rrc_ratio

        images = mnist.data.float().unsqueeze(1) / 255.0
        images = (images - MNIST_MEAN[0]) / MNIST_STD[0]

        if cfg.trigger_type == 'hadamard':
            hadamard_triggers = get_hadamard_triggers(n_classes=10, intensity=self.intensity)
            for idx in range(len(images)):
                bd_class = self.bd_classes[idx].item()
                images[idx, 0, 0, :16] = hadamard_triggers[bd_class]
        else:
            for idx in range(len(images)):
                bd_class = self.bd_classes[idx].item()
                trigger = torch.full((10,), -self.intensity / 9)
                trigger[bd_class] = self.intensity
                images[idx, 0, 0, :10] = trigger

        if subset_indices is not None:
            images = images[subset_indices]
            self.labels = mnist.targets[subset_indices].to(device)
            self.bd_classes = self.bd_classes[subset_indices]
        else:
            self.labels = mnist.targets.to(device)

        images = images.to(device)

        # Apply ZCA whitening after inserting triggers and selecting the subset.
        # Reuse training statistics when zca_stats is provided, as for test or linear-probe data.
        # Otherwise, compute statistics from this dataset, as for training data.
        self.zca_stats = None
        if cfg.get('whitening', False):
            if zca_stats is None:
                zca_stats = compute_zca(images, eps=cfg.get('whitening_eps', 1e-2))
                print(f"[ZCA] computed from {images.size(0)} samples "
                      f"(eps={cfg.get('whitening_eps', 1e-2)})")
            zca_mean, zca_W = zca_stats
            images = apply_zca(images, zca_mean.to(images.device), zca_W.to(images.device))
            self.zca_stats = zca_stats

        self.images = images

    def _apply_rrc(self, img):
        """Apply RRC directly to GPU tensors."""
        i, j, h, w = RandomResizedCrop.get_params(
            img, scale=self.rrc_scale, ratio=self.rrc_ratio
        )
        return TF.resized_crop(img, i, j, h, w, [self.img_size, self.img_size])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if self.mode == 'ssl':
            # Apply RRC.
            view1 = self._apply_rrc(img)
            view2 = self._apply_rrc(img)

            # Add noise.
            if self.noise_std > 0:
                view1 = view1 + torch.randn_like(view1) * self.noise_std
                view2 = view2 + torch.randn_like(view2) * self.noise_std

            return view1, view2, label
        else:
            return img, label


class Encoder(nn.Module):
    def __init__(self, hidden_dims=[256, 256, 256, 256], proj_dim=32):
        super().__init__()
        layers = []
        in_dim = 28 * 28
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        self.feature_dim = hidden_dims[-1]

        self.projector = nn.Linear(self.feature_dim, proj_dim)

    def forward(self, x, use_projector=True):
        x = x.view(x.size(0), -1)
        features = self.backbone(x)
        if use_projector:
            return self.projector(features)
        return features


class BarlowTwinsLoss(nn.Module):
    def __init__(self, proj_dim, lambd=0.005):
        super().__init__()
        self.lambd = lambd
        self.bn = nn.BatchNorm1d(proj_dim, affine=False)

    def forward(self, z1, z2):
        batch_size = z1.size(0)
        z1 = self.bn(z1)
        z2 = self.bn(z2)
        c = torch.mm(z1.T, z2) / batch_size

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return c, loss

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def train_linear_probe(model, train_loader, epochs=30):
    """Train linear probe on frozen encoder and save full model"""
    model.eval()

    classifier = nn.Linear(model.feature_dim, 10).to(device)
    optimizer = torch.optim.Adam(classifier.parameters())

    print("\nTraining linear probe...")
    for epoch in tqdm(range(epochs)):
        classifier.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                features = model(images, use_projector=False)
            logits = classifier(features)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: Loss = {total_loss / len(train_loader):.4f}")

    return classifier


def test_with_random_backdoor(model, test_loader, classifier):
    """Test with random backdoor (spuriosity=0)"""

    model.eval()
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            features = model(images, use_projector=False)
            logits = classifier(features)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def train(train_loader, cfg, csv_path, train_non_aug_loader, test_loader):
    intensity = cfg.intensity
    max_epoch = cfg.max_epoch
    print(f"\n{'=' * 50}")
    print(f"Training with intensity={intensity:.4f}, epochs={max_epoch}")
    print(f"{'=' * 50}")

    model = Encoder(proj_dim=128).to(device)

    if cfg.scaling != -1:
        cumulative1, cumulative2 = [], []
        for i in range(6):
            view1, view2, _ = next(iter(train_loader))
            cumulative1.append(view1)
            cumulative2.append(view2)
        cumulative1, cumulative2 = torch.cat(cumulative1, dim=0), torch.cat(cumulative2, dim=0)

        def evaluate(m, data):
            view1, view2 = data
            eigvals = common.compute_matrix_eigenvals(m, view1, view2, embeddings=True)
            return eigvals.max()

        weights_init_scale, top_c_eigval = common.find_sufficiently_small_init(model=model, eval_fn=evaluate,
                                                                               eval_target=cfg.scaling,
                                                                               data=(cumulative1, cumulative2),
                                                                               iterations=100)
        print(f"{weights_init_scale}, {top_c_eigval}")
        common.scale_weights(model, weights_init_scale, scale_bias=True)

    # loss_fn = BarlowTwinsLoss(proj_dim=32, lambd=5e-3).to(device)
    loss_fn = BarlowTwinsLoss(proj_dim=128, lambd=5e-3).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-6)
    # optimizer = torch.optim.Adam(model.parameters())

    step = 0
    for epoch in tqdm(range(max_epoch)):
        model.train()
        total_loss = 0
        for view1, view2, _ in train_loader:
            view1, view2 = view1.to(device), view2.to(device)
            z1, z2 = model(view1), model(view2)
            _, loss = loss_fn(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if step == cfg.max_step:
                return model, step
            if cfg.test_step_freq > 0 and (step + 1) % cfg.test_step_freq == 0:
                classifier = train_linear_probe(model, train_non_aug_loader, epochs=30)
                test_acc = test_with_random_backdoor(model, test_loader, classifier)
                write_csv(csv_path, step, cfg, test_acc)
                print(f"Test accuracy {epoch}: {test_acc:.2f}%")
            step += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Loss = {total_loss / len(train_loader):.4f}")
        if (epoch + 1) in cfg.test_epoch:
            classifier = train_linear_probe(model, train_non_aug_loader, epochs=30)
            test_acc = test_with_random_backdoor(model, test_loader, classifier)
            write_csv(csv_path, step, cfg, test_acc)
            print(f"Test accuracy {epoch}: {test_acc:.2f}%")

    return model, step


@hydra.main(version_base=None, config_path="configs", config_name="conf")
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    os.makedirs(f"result/{cfg.spuriosity}/{cfg.seed}_seed", exist_ok=True)
    if cfg.max_step != -1:
        csv_path = os.path.join(
            f"result/{cfg.spuriosity}/{cfg.seed}_seed/{cfg.scaling}_{cfg.max_step}step_{cfg.subset_ratio}subset.csv")
    else:
        csv_path = os.path.join(
            f"result/{cfg.spuriosity}/{cfg.seed}_seed/{cfg.scaling}_{cfg.max_epoch}step_{cfg.subset_ratio}subset.csv")
    directory = '/home/dataset/mnist_backdoor/'

    image_count = 60000
    n = int(image_count * cfg.subset_ratio)
    indices = torch.randperm(image_count)[:n]

    # Compute ZCA statistics on the training set when cfg.whitening is True.
    train_dataset = BackdoorMNIST(bd_path=f'{directory}/train_spur{cfg.spuriosity}_seed42.pt',
                                  train=True, mode='ssl', subset_indices=indices, cfg=cfg)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    # When whitening is disabled, None also passes through the datasets below.
    zca_stats = train_dataset.zca_stats

    train_linear_dataset = BackdoorMNIST(bd_path=f'{directory}/train_spur{cfg.spuriosity}_seed42.pt',
                                         train=True, mode='linear', subset_indices=indices,
                                         cfg=cfg, zca_stats=zca_stats)
    train_linear_loader = DataLoader(train_linear_dataset, batch_size=128, shuffle=True, num_workers=0)

    # image_count = 10000
    # n = int(image_count * cfg.subset_ratio)
    # indices = torch.randperm(image_count)[:n]
    test_dataset = BackdoorMNIST(bd_path=f'{directory}/test_spur{cfg.spuriosity}_seed42.pt',
                                 train=False, mode='eval', cfg=cfg, zca_stats=zca_stats)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0)

    # SSL training
    model, step = train(train_loader, cfg, csv_path, train_linear_loader, test_loader)

    # Linear probe training
    classifier = train_linear_probe(model, train_linear_loader)
    test_acc = test_with_random_backdoor(model, test_loader, classifier)

    # Test with random backdoor
    print(f"Test accuracy (random backdoor): {test_acc:.2f}%")

    write_csv(csv_path, step, cfg, test_acc)


def write_csv(csv_path, step, cfg, test_acc):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['step', 'intensity', 'test_acc'])
        writer.writerow([step, cfg.intensity, test_acc])
    print(f"Results appended to {csv_path}")


if __name__ == "__main__":
    main()