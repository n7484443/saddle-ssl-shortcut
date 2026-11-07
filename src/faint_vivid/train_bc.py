"""Faint/Vivid optimized single-file training.

Edit Config below for data_dir, output_dir and seed. Run: python train_bc.py
Requires CUDA PyTorch, torchvision, numpy, Pillow and tqdm; original RGB48 data/manifest files.
Constructor-default Conv/Linear initialization, followed by the existing six-batch scale calibration.
Scientific settings: lambda0.028671875, LR1.6, scaling5e-8, warmup2000, steps30000, probe500.
Configuration and normalization are embedded. Optional launch_config.json overrides are supported.
"""

import copy
import csv
import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T
from tqdm import tqdm


# Configuration for the noorth 30k run. Set data_dir, output_dir, and seed here.
ROOT = Path(__file__).resolve().parent
device = torch.device("cuda")


@dataclass
class Config:
    data_dir: str = 'data/faint_vivid'
    output_dir: str = str(ROOT / "runs")
    seed: int = 23
    lambd: float = 0.028671875
    loss_mode: str = 'per_view_batch_normalized'
    optimizer: str = 'lars'
    scaling: float | None = 5e-08
    lr: float = 1.6
    warmup_steps: int = 2000
    bias_bn_lr: float = 0.0048
    momentum: float = 0.9
    trust_coefficient: float = 0.001
    optimizer_eps: float = 1e-08
    weight_decay: float = 1e-06
    batch_size: int = 256  # faint + vivid images per view
    steps: int = 30000
    probe_every: int = 500
    probe_epochs: int = 35
    probe_lr: float = 0.001
    rrc_min: float = 0.3
    affine_fill: tuple[float, float, float] = field(default_factory=lambda: [0.507477343082428, 0.5259000658988953, 0.5021345019340515])
    affine_translate: tuple[float, float] = field(default_factory=lambda: [0.05, 0.05])
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.0
    color_jitter_hue: float = 0.0
    fixed_batch: bool = False
    workers: int = 4
    # Reset biases to zero and reapply weight scaling after loading a checkpoint.
    initial_checkpoint: str | None = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FaintVividDataset(Dataset):
    def __init__(self, cfg, split, augment=False, stats=None):
        directory = Path(cfg.data_dir) / split
        with (directory / "manifest.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.paths = [[directory / row[f"cond{i}_image"] for i in range(2)] for row in rows]
        self.labels = torch.tensor([[int(row[f"cond{i}_label"]) for i in range(2)] for row in rows])
        self.conditions = torch.tensor([[int(row[f"cond{i}_p"]) for i in range(2)] for row in rows])
        if not rows or set(self.conditions.flatten().tolist()) != {10, 45}:
            raise ValueError("Expected original faint/vivid manifest with p=10 and p=45.")
        self.augment = augment
        # Cache only decoded immutable RGB bytes. All RNG-using views/collation remain unchanged.
        # The parent populates one CPU shared storage before any DataLoader workers start.
        self.rgb_cache = torch.empty((len(self.paths), 2, 48, 48, 3), dtype=torch.uint8, device="cpu").share_memory_()
        for row_index, pair in enumerate(self.paths):
            for condition_index, path in enumerate(pair):
                image = self.read_image(path)
                if image.mode != "RGB" or image.size != (48, 48):
                    raise ValueError("Decoded cache requires the original RGB48 dataset.")
                pixels = np.array(image, dtype=np.uint8, copy=True)
                self.rgb_cache[row_index, condition_index].copy_(torch.from_numpy(pixels))

        if stats is None:
            cached = {'mean': [0.507477343082428, 0.5259000658988953, 0.5021345019340515], 'std': [0.023013902828097343, 0.17264536023139954, 0.0015600168844684958]}
            stats = torch.tensor(cached["mean"]), torch.tensor(cached["std"])
        self.mean, self.std = stats
        self.normalize = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        self.view = T.Compose([
            T.RandomResizedCrop(48, scale=(cfg.rrc_min, 1.0)),
            T.ToTensor(),
            T.RandomAffine(degrees=10, translate=cfg.affine_translate, scale=(0.95, 1.05), shear=5, fill=cfg.affine_fill),
            T.ColorJitter(brightness=cfg.color_jitter_brightness, contrast=cfg.color_jitter_contrast,
                          saturation=cfg.color_jitter_saturation, hue=cfg.color_jitter_hue)
            if any((cfg.color_jitter_brightness, cfg.color_jitter_contrast,
                    cfg.color_jitter_saturation, cfg.color_jitter_hue)) else T.Identity(),
            T.Normalize(self.mean, self.std),
        ])

    @staticmethod
    def read_image(path):
        with Image.open(path) as image:
            return image.convert("RGB")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        images = [Image.fromarray(self.rgb_cache[index, condition_index].numpy()) for condition_index in range(2)]
        if self.augment:
            return (torch.stack([self.view(x) for x in images]),
                    torch.stack([self.view(x) for x in images]))
        return (torch.stack([self.normalize(x) for x in images]),
                self.labels[index], self.conditions[index])


def collate_conditions(batch):
    values = [torch.stack(items).flatten(0, 1) for items in zip(*batch)]
    order = torch.randperm(len(values[0]))
    return tuple(value[order] for value in values)


def make_loaders(cfg):
    train = FaintVividDataset(cfg, "train", augment=True)
    train_noaug = copy.copy(train)
    train_noaug.augment = False
    test = FaintVividDataset(cfg, "test", stats=(train.mean, train.std))
    options = dict(num_workers=cfg.workers, pin_memory=True,
                   persistent_workers=cfg.workers > 0, collate_fn=collate_conditions)
    return (
        DataLoader(train, batch_size=cfg.batch_size // 2, shuffle=True, drop_last=True, **options),
        DataLoader(train_noaug, batch_size=cfg.batch_size // 2, drop_last=True, **options),
        DataLoader(test, batch_size=1024, **options),
    )


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 128, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.01), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.01), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.01), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(512 * 6 * 6, 64, bias=True),
        )

    def forward(self, x):
        return self.layers(x)


class SSLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.projector = nn.Linear(64, 32, bias=True)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.projector(self.encode(x))


class LARS(Optimizer):
    def __init__(self, params, lr, momentum=0.9, weight_decay=1e-6,
                 trust_coefficient=1e-3, eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        trust_coefficient=trust_coefficient, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                update = parameter.grad
                if parameter.ndim > 1:
                    parameter_norm = torch.linalg.vector_norm(parameter)
                    update_norm = torch.linalg.vector_norm(update)
                    denominator = update_norm + group["weight_decay"] * parameter_norm
                    trust_ratio = torch.where(
                        (parameter_norm > 0) & (denominator > 0),
                        group["trust_coefficient"] * parameter_norm / (denominator + group["eps"]),
                        torch.ones_like(parameter_norm),
                    )
                    update = update.add(parameter, alpha=group["weight_decay"]).mul(trust_ratio)
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                buffer = state["momentum_buffer"]
                buffer.mul_(group["momentum"]).add_(update)
                parameter.add_(buffer, alpha=-group["lr"])


def joint_mean_symmetric_c(z1, z2):
    mu = (z1.mean(dim=0) + z2.mean(dim=0)) / 2
    y1, y2 = z1 - mu, z2 - mu
    cross = y1.T @ y2
    return (cross + cross.T) / (2 * len(y1))


class BTLoss(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd
        self.bn = nn.BatchNorm1d(32, affine=False, track_running_stats=False, eps=1e-16)

    def forward(self, z1, z2):
        u1 = self.bn(z1)
        u2 = self.bn(z2)
        cross = u1.T @ u2
        c = (cross + cross.T) / (2 * len(z1))
        on_diag = (torch.diagonal(c) - 1).pow(2).sum()
        n = c.shape[0]
        off_diag = c.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten().pow(2).sum()
        return c, on_diag + self.lambd * off_diag


@torch.no_grad()
def initialize(model, x1, x2, cfg):
    model(x1[:2])  # Preserve the original initialization warm-up.
    if cfg.initial_checkpoint:
        saved = torch.load(cfg.initial_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
    for name, parameter in model.named_parameters():
        if name.endswith("bias"):
            parameter.zero_()

    # Reuse the original six calibration batches, with the same concatenated training-sized inputs; small CNN has no BN.
    if len(x1) != len(x2) or len(x1) % cfg.batch_size:
        raise ValueError("Calibration views must contain complete training-sized batches.")
    batches = list(zip(x1.split(cfg.batch_size), x2.split(cfg.batch_size)))

    def calibration_matrices():
        centered_sum, raw_sum = None, None
        for view1, view2 in batches:
            z1, z2 = model(torch.cat([view1, view2])).chunk(2)
            centered = joint_mean_symmetric_c(z1, z2)
            raw = (z1.T @ z2 + z2.T @ z1) / (2 * len(z1))
            centered_sum = centered if centered_sum is None else centered_sum + centered
            raw_sum = raw if raw_sum is None else raw_sum + raw
        return centered_sum / len(batches), raw_sum / len(batches)

    definition = {
        "batch_count": len(batches), "per_view_batch_size": cfg.batch_size,
        "bn_context": "모델 encoder/projector에는 BN이 없고 SSL criterion에만 affineFalse/track_running_statsFalse/eps1e-16 BN32가 있다. 초기 교정은 criterion을 거치지 않는 기존 모델의 합친512 입력·동일6batch joint-centered RAW target0.001 solver/회당 구성을 그대로 유지하며 raw_top은 별도 uncentered 값이다. 별도 진단 forward를 추가하지 않고 탐색 반복수나 초기 정확도·성공을 보장하지 않는다.",
        "scaling_top_definition": "top eigenvalue of mean of per-batch joint-mean-centered symmetric cross-view C",
        "raw_top_definition": "top eigenvalue of mean of per-batch uncentered symmetric cross-view C from the same final forwards",
        "target": cfg.scaling,
        "matrix_initialization": "Conv2d_Linear_constructor_default_initialization_no_explicit_orthogonal_then_existing_six_batch_scaling_calibration",
    }
    if cfg.scaling is None:
        centered, raw = calibration_matrices()
        return {"scaling_top": torch.linalg.eigvalsh(centered)[-1].item(),
                "raw_top": torch.linalg.eigvalsh(raw)[-1].item(),
                "calibration_definition": definition, "weight_factor": 1.0}
    original = copy.deepcopy(model.state_dict())
    factor, exponent, previous_direction = 1.0, 0, None
    best_factor, best_value = 1.0, None
    for _ in range(100):
        model.load_state_dict({name: value * factor if name.endswith("weight") else value for name, value in original.items()})
        centered, _ = calibration_matrices()
        value = torch.linalg.eigvalsh(centered)[-1].item()
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"Invalid initialization eigenvalue: {value}")
        better = best_value is None or abs(math.log10(cfg.scaling / value)) < abs(math.log10(cfg.scaling / best_value))
        if better:
            best_factor, best_value = factor, value
        if abs(cfg.scaling - best_value) < 1e-8:
            break
        direction = 1 if cfg.scaling > best_value else -1
        if not better or (previous_direction is not None and direction != previous_direction):
            exponent += 1
        factor = best_factor * 10 ** (direction * 0.5 ** exponent)
        previous_direction = direction
    model.load_state_dict({name: value * best_factor if name.endswith("weight") else value for name, value in original.items()})
    # These existing final calibration forwards supply both reported matrices; no diagnostic pass.
    centered, raw = calibration_matrices()
    scaling_top = torch.linalg.eigvalsh(centered)[-1].item()
    raw_top = torch.linalg.eigvalsh(raw)[-1].item()
    if abs(scaling_top - cfg.scaling) > max(5e-8, cfg.scaling * 1e-3):
        print(f"Scaling target not reached; proceeding: target={cfg.scaling}, actual={scaling_top}", flush=True)
    return {"scaling_top": scaling_top, "raw_top": raw_top,
            "calibration_definition": definition, "weight_factor": best_factor}


@torch.no_grad()
def extract_features(model, loader):
    features, labels, conditions = [], [], []
    for images, y, p in loader:
        features.append(model.encode(images.to(device, non_blocking=True)))
        labels.append(y.to(device, non_blocking=True))
        conditions.append(p.to(device, non_blocking=True))
    return torch.cat(features), torch.cat(labels), torch.cat(conditions)


def linear_probe(model, head, initial_head, train, test, cfg):
    """Shared digit head on raw features. Restart from the same initialization at each evaluation."""
    model.eval()
    x_train, y_train, _ = extract_features(model, train)
    x_test, y_test, p_test = extract_features(model, test)
    head.load_state_dict(initial_head)
    head.train()
    optimizer = torch.optim.Adam(head.parameters(), lr=cfg.probe_lr)
    generator = torch.Generator(device=device).manual_seed(0)
    order = torch.randperm(len(x_train), generator=generator, device=device)
    x_ordered, y_ordered = x_train[order], y_train[order]
    probe_batches = tuple(zip(x_ordered.split(cfg.batch_size), y_ordered.split(cfg.batch_size)))
    for _ in range(cfg.probe_epochs):
        for batch_x, batch_y in probe_batches:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    head.eval()
    with torch.no_grad():
        correct = head(x_test).argmax(1) == y_test
        accuracies = [100 * correct[p_test == p].sum().item() / (p_test == p).sum().item()
                      for p in (10, 45)]
    model.train()
    return accuracies


def main(cfg):
    if cfg.batch_size % 2 or cfg.steps <= 0 or cfg.probe_every <= 0:
        raise ValueError("Use an even batch size and positive steps/probe_every.")
    torch.set_num_threads(4)
    set_seed(cfg.seed)
    train, train_noaug, test = make_loaders(cfg)
    if cfg.scaling is not None and not cfg.fixed_batch and len(train) < 6:
        raise ValueError("Scaling requires at least six full training batches.")
    set_seed(cfg.seed)
    output = Path(cfg.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    (output / "normalization.json").write_text(json.dumps({
        "mean": train.dataset.mean.tolist(), "std": train.dataset.std.tolist(),
    }, indent=2), encoding="utf-8")

    model = SSLModel().to(device)
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)) and module.bias is not None:
            module.weight.requires_grad_(False)
            module.bias.requires_grad_(False)
            module.bias.zero_()
    batches = iter(train)
    views = [next(batches) for _ in range(1 if cfg.fixed_batch or cfg.scaling is None else 6)]
    x1_init, x2_init = [torch.cat([v[i].to(device, non_blocking=True) for v in views]) for i in range(2)]
    calibration = initialize(model, x1_init, x2_init, cfg)
    top = calibration["raw_top"]
    scaling_top = calibration["scaling_top"]
    training_batches = [(x1_init, x2_init)] if cfg.fixed_batch else train
    if cfg.fixed_batch:
        torch.save({"x1": x1_init.cpu(), "x2": x2_init.cpu()}, output / "fixed_views.pt")
    print(f"Initial raw top eigenvalue: {top:.8g}; scaling top eigenvalue: {scaling_top:.8g}; output: {output}")
    criterion = BTLoss(cfg.lambd).to(device)
    weight_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad and parameter.ndim > 1]
    bias_bn_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad and parameter.ndim == 1]
    parameter_groups = [{"params": weight_parameters, "lr": cfg.lr,
                         "weight_decay": cfg.weight_decay}]
    if bias_bn_parameters:
        parameter_groups.append({"params": bias_bn_parameters, "lr": cfg.bias_bn_lr,
                                 "weight_decay": 0.0})
    optimizer = LARS(parameter_groups, lr=cfg.lr, momentum=cfg.momentum,
                     weight_decay=cfg.weight_decay, trust_coefficient=cfg.trust_coefficient,
                     eps=cfg.optimizer_eps)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    head = nn.Linear(64, 10).to(device)
    initial_head = copy.deepcopy(head.state_dict())
    torch.save({"model": model.state_dict(), "step": 0, "raw_top": top, "scaling_top": scaling_top, "calibration_definition": calibration["calibration_definition"]}, output / "initial.pt")

    step, state = 0, "completed"
    # Accumulate on the GPU; inspect only at the existing logging boundaries.
    finite_since_log = torch.ones((), dtype=torch.bool, device=device)
    stop_request_path = ROOT / "STOP_REQUESTED"
    with (output / "loss.csv").open("w", newline="") as lf, (output / "accuracy.csv").open("w", newline="") as af:
        loss_log, accuracy_log = csv.writer(lf), csv.writer(af)
        # Eigenvalues of loss-only per-view batch-normalized symmetric C32 before the logged update; initialization scaling_top remains joint-centered RAW and raw_top remains separately uncentered.
        loss_log.writerow(["step", "loss", "c_eigenvalue_1", "c_eigenvalue_2", "lr"])
        accuracy_log.writerow(["step", "faint", "vivid"])
        try:
            faint, vivid = linear_probe(model, head, initial_head, train_noaug, test, cfg)
            accuracy_log.writerow([0, faint, vivid])
            af.flush()
            if faint >= 80 or vivid >= 80:
                state = "invalid_initial_ge80"
                cfg.steps = 0
            with tqdm(total=cfg.steps, desc="SSL") as bar:
                while step < cfg.steps:
                    for x1, x2 in training_batches:
                        if stop_request_path.exists():
                            state = "early_stopped_requested"
                            cfg.steps = step
                            break
                        model.train()
                        optimizer.zero_grad(set_to_none=True)
                        x1_gpu = x1.to(device, non_blocking=True)
                        x2_gpu = x2.to(device, non_blocking=True)
                        z1, z2 = model(torch.cat([x1_gpu, x2_gpu])).chunk(2)
                        c, loss = criterion(z1, z2)
                        finite_since_log.logical_and_(torch.isfinite(loss.detach()))
                        log_step = (step + 1) % cfg.probe_every == 0 or step + 1 == cfg.steps
                        if log_step:
                            with torch.no_grad():
                                # This boundary already transfers eigenvalues and loss to the host.
                                if not finite_since_log.item():
                                    raise RuntimeError(f"Nonfinite loss in interval ending at step {step + 1}")
                                finite_since_log.fill_(True)
                                eig1, eig2 = torch.linalg.eigvalsh(c.detach())[-2:].flip(0).tolist()
                        loss.backward()
                        multiplier = min((step + 1) / cfg.warmup_steps, 1.0) if cfg.warmup_steps > 0 else 1.0
                        for group in optimizer.param_groups:
                            group["lr"] = group["initial_lr"] * multiplier
                        optimizer.step()
                        step += 1
                        if log_step:
                            loss_log.writerow([step, loss.item(), eig1, eig2, optimizer.param_groups[0]["lr"]])
                            lf.flush()
                        bar.update()
                        if step % cfg.probe_every == 0 or step == cfg.steps:
                            faint, vivid = linear_probe(model, head, initial_head, train_noaug, test, cfg)
                            accuracy_log.writerow([step, faint, vivid])
                            af.flush()
                            tqdm.write(f"step={step}: faint={faint:.2f}%, vivid={vivid:.2f}%, "
                                       f"C eig1={eig1:.6g}, eig2={eig2:.6g}")
                        if step % 5000 == 0:
                            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                        "step": step}, output / f"step_{step:06d}.pt")
                        if step >= cfg.steps:
                            break
        except KeyboardInterrupt:
            state = "interrupted"
            print("Stopped; saving current weights.")
        except Exception:
            state = "failed"
            raise
        finally:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step, "state": state}, output / "last.pt")


if __name__ == "__main__":
    config_path = ROOT / "launch_config.json"
    overrides = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    main(Config(**overrides))



