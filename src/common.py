from typing import Tuple

import torch
import torch.nn.functional as F
import numpy as np
import random
import torch.nn as nn
from contextlib import contextmanager

import os
from tqdm import tqdm
import math


def set_seed(val):
    torch.manual_seed(val)
    torch.cuda.manual_seed(val)
    torch.cuda.manual_seed_all(val)  # for multi gpu
    np.random.seed(val)
    random.seed(val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def random_resized_crop_1d(x, crop_ratio_min, crop_ratio_max):
    batch_size = x.shape[0]
    device = x.device
    seq_len = x.shape[1]

    crop_ratios = torch.rand(batch_size, device=device) * (crop_ratio_max - crop_ratio_min) + crop_ratio_min
    crop_lengths = (seq_len * crop_ratios).round()
    crop_lengths = torch.clamp(crop_lengths, min=1, max=seq_len)

    max_starts = seq_len - crop_lengths
    starts = (torch.rand(batch_size, device=device) * max_starts.float()).long()

    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)

    # Normalized coordinates from 0 to crop_lengths-1
    normalized_coords = torch.linspace(0, 1, seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    crop_coords = starts.unsqueeze(1) + normalized_coords * (crop_lengths.unsqueeze(1) - 1).float()
    crop_coords = crop_coords.clamp(0, seq_len - 1)

    # Linear interpolation
    crop_coords_floor = crop_coords.floor().long()
    crop_coords_ceil = (crop_coords_floor + 1).clamp(max=seq_len - 1)
    weight = (crop_coords - crop_coords_floor.float()).unsqueeze(-1)

    if x.dim() == 2:
        x = x.unsqueeze(-1)
        squeeze_last = True
    else:
        squeeze_last = False

    x_floor = x[batch_indices, crop_coords_floor]
    x_ceil = x[batch_indices, crop_coords_ceil]
    x_cropped = x_floor * (1 - weight) + x_ceil * weight

    if squeeze_last:
        x_cropped = x_cropped.squeeze(-1)

    return x_cropped


def create_segment_vectors(sizes):
    total_length = sum(sizes)
    cumsum = torch.cumsum(torch.tensor(sizes), dim=-1)

    result = torch.zeros((len(sizes), total_length))

    for i in range(len(sizes)):
        start_idx = cumsum[i - 1] if i > 0 else 0
        end_idx = cumsum[i]

        result[i, start_idx:end_idx] = 1

    return result


def calculate_gamma(x0, x1):
    gamma = torch.bmm(x0.unsqueeze(dim=2), x1.unsqueeze(dim=1))  # b x n x n
    gamma = gamma + gamma.permute(0, 2, 1)
    gamma = torch.mean(gamma, dim=0).squeeze() / 2  # n x n
    gamma_l, gamma_v = torch.linalg.eig(gamma)
    gamma_l = gamma_l.real.cpu()
    gamma_v = gamma_v.real.cpu()
    return gamma, gamma_l, gamma_v


def sample(batch, sizes, device, mul=0.01):
    x_out = torch.zeros((batch, 0), device=device)
    for size in sizes:
        rand = torch.rand((batch, 1), device=device).round() * 2 - 1
        x_out = torch.cat([x_out, rand.repeat(1, size)], dim=-1)

    epsilon1 = mul * torch.randn_like(x_out)
    epsilon2 = mul * torch.randn_like(x_out)

    x1 = x_out + epsilon1
    x2 = x_out + epsilon2

    return x1, x2


class BTLoss(nn.Module):
    def __init__(self, use_batchnorm, lambd=1, in_feature=0):
        super().__init__()
        if use_batchnorm:
            assert (in_feature > 0)
            self.bn = nn.BatchNorm1d(in_feature, affine=False)
        else:
            self.bn = nn.Identity()
        self.lambd = lambd

    def forward(self, x0, x1):
        c = self.bn(x0).T @ self.bn(x1)
        c = c + c.T
        c.div_(x0.shape[0] * 2)
        output = c.clone()
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return output, loss


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class SimCLRLoss(nn.Module):
    def __init__(self, batch_size, temperature, use_batchnorm=False, in_feature=None):
        super().__init__()
        if use_batchnorm:
            self.bn = nn.BatchNorm1d(in_feature, affine=False)
        else:
            self.bn = nn.Identity()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    # Mask for inner products between negative samples in the loss denominator.
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)

        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, x0, x1):
        with torch.inference_mode():
            c = self.bn(x0).T @ self.bn(x1)
            c = c + c.T
            c.div_(x0.shape[0] * 2)

        N = 2 * self.batch_size

        z = torch.cat((x0, x1), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.from_numpy(np.array([0] * N)).reshape(-1).to(positive_samples.device).long()

        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return c, loss


class VICRegLoss(nn.Module):
    def __init__(
            self,
            inv_coeff: float = 25.0,
            var_coeff: float = 25.0,
            cov_coeff: float = 1.0,
            gamma: float = 1.0,
    ):
        super().__init__()
        self.inv_coeff = inv_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma
        self.bn = nn.Identity()

    def forward(self, x0: torch.Tensor, x1: torch.Tensor):
        metrics = dict()
        metrics["inv-loss"] = self.inv_coeff * self.representation_loss(x0, x1)
        metrics["var-loss"] = (
                self.var_coeff
                * (self.variance_loss(x0, self.gamma) + self.variance_loss(x1, self.gamma))
                / 2
        )
        metrics["cov-loss"] = (
                self.cov_coeff * (self.covariance_loss(x0) + self.covariance_loss(x1)) / 2
        )

        with torch.inference_mode():
            c = self.bn(x0).T @ self.bn(x1)
            c = c + c.T
            c.div_(x0.shape[0] * 2)
        return c, sum(metrics.values())

    @staticmethod
    def representation_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x, y)

    @staticmethod
    def variance_loss(x: torch.Tensor, gamma: float) -> torch.Tensor:
        x = x - x.mean(dim=0)
        std = torch.sqrt(x.var(dim=0) + 1e-6)
        var_loss = F.relu(gamma - std).mean()
        return var_loss

    @staticmethod
    def covariance_loss(x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=0)
        cov = (x.T @ x) / (x.shape[0] - 1)
        cov_loss = cov.fill_diagonal_(0.0).pow(2).sum() / x.shape[1]
        return cov_loss


class SIGReg(nn.Module):
    def __init__(self, knots=17, num_random_features=256):
        super().__init__()
        self.num_random_features = num_random_features

        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)

        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt

        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(
            proj.size(-1),
            self.num_random_features,
            device=proj.device,
            dtype=proj.dtype,
        )
        A = F.normalize(A, p=2, dim=0)

        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(0) - self.phi).square() + x_t.sin().mean(0).square()
        statistic = (err @ self.weights) * proj.size(0)
        return statistic.mean()


class LeJEPALoss(nn.Module):
    def __init__(self, lamb=0.02, knots=17):
        super().__init__()
        self.lamb = lamb
        self.sigreg = SIGReg(knots=knots)

    def forward(self, x0, x1):
        proj = torch.stack([x0, x1], dim=0)  # [2, N, D]

        inv_loss = (proj.mean(0) - proj).square().mean()
        sigreg_loss = (self.sigreg(x0) + self.sigreg(x1)) / 2
        loss = sigreg_loss * self.lamb + inv_loss * (1 - self.lamb)

        return None, loss


@contextmanager
def scaled_model(model, scale, scale_bias):
    assert scale != 0
    try:
        scale_weights(model, scale, scale_bias)
        yield model
    finally:
        scale_weights(model, 1 / scale, scale_bias)


@torch.no_grad()
def scale_weights(model, scale: float = 1, scale_bias=False, scale_bn=True) -> None:
    for name, param in model.named_parameters():
        is_bn = "bn" in name or "norm" in name

        if is_bn and not scale_bn:
            continue

        if "weight" in name and scale != 1:
            param.copy_(param * scale)
        elif scale_bias and "bias" in name and scale != 1:
            param.copy_(param * scale)

@torch.no_grad()
def small_init_projector_readout(model, scale: float = 0.1):
    projector = model.projector

    last_linear = None
    for module in reversed(list(projector.modules())):
        if isinstance(module, torch.nn.Linear):
            last_linear = module
            break

    if last_linear is None:
        raise ValueError("No Linear layer found in model.projector")

    last_linear.weight.mul_(scale)

    if last_linear.bias is not None:
        last_linear.bias.zero_()


from torch.nn.modules.batchnorm import _BatchNorm
from typing import Literal

class ScaleOutput(nn.Module):
    """
    Wraps a module and multiplies its output by a fixed scalar.

    Used for:
        residual branch: F(x) -> alpha * F(x)
    """
    def __init__(self, module: nn.Module, scale: float):
        super().__init__()
        self.module = module
        self.register_buffer("scale", torch.tensor(float(scale)))

    def forward(self, *args, **kwargs):
        return self.scale * self.module(*args, **kwargs)


def _is_norm_module(module: nn.Module) -> bool:
    return isinstance(
        module,
        (
            _BatchNorm,
            nn.GroupNorm,
            nn.LayerNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
        ),
    )


def _find_residual_tail_norm_name(block: nn.Module) -> str | None:
    """
    Finds the final normalization layer of a torchvision-style residual branch.

    BasicBlock:
        conv1 -> bn1 -> relu -> conv2 -> bn2
        target: bn2

    Bottleneck:
        conv1 -> bn1 -> conv2 -> bn2 -> conv3 -> bn3
        target: bn3
    """
    candidate_names = ("bn3", "norm3", "bn2", "norm2")

    for name in candidate_names:
        if not hasattr(block, name):
            continue

        module = getattr(block, name)

        if isinstance(module, ScaleOutput):
            return None

        if _is_norm_module(module):
            return name

    return None


@torch.no_grad()
def apply_residual_scaling(
    model: nn.Module,
    scale: float = 1.0,
    mode: Literal["fixed", "init"] = "fixed",
    verbose: bool = True,
) -> list[str]:
    """
    Applies residual-branch scaling to ResNet-style blocks.

    This replaces the old global weight scaling.

    mode="fixed":
        Replaces the final BN/norm in each residual branch with:
            ScaleOutput(final_norm, scale)

        Forward becomes:
            out = identity + scale * F(x)

        This is the cleaner diagnostic setting because alpha stays fixed.

    mode="init":
        Multiplies only the final BN/norm gamma by scale:
            final_norm.weight *= scale

        Forward initially behaves like:
            out = identity + scale * F(x)

        But the gamma parameter can grow during training.

    Important:
        This function does NOT scale conv weights.
        This function does NOT scale all BN layers.
        This function does NOT scale bias terms.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")

    if mode not in {"fixed", "init"}:
        raise ValueError(f"mode must be 'fixed' or 'init', got {mode}")

    scaled_modules: list[str] = []

    # Use list(...) because we may replace modules while iterating.
    for block_name, block in list(model.named_modules()):
        tail_norm_name = _find_residual_tail_norm_name(block)

        if tail_norm_name is None:
            continue

        tail_norm = getattr(block, tail_norm_name)

        # Heuristic: only touch actual residual blocks.
        # torchvision BasicBlock/Bottleneck have downsample/stride and final bn2/bn3.
        is_residual_like = (
            hasattr(block, "downsample")
            or "BasicBlock" in block.__class__.__name__
            or "Bottleneck" in block.__class__.__name__
            or "Residual" in block.__class__.__name__
        )

        if not is_residual_like:
            continue

        full_name = f"{block_name}.{tail_norm_name}" if block_name else tail_norm_name

        if mode == "fixed":
            setattr(block, tail_norm_name, ScaleOutput(tail_norm, scale))
            scaled_modules.append(full_name)

        elif mode == "init":
            if not hasattr(tail_norm, "weight") or tail_norm.weight is None:
                continue

            tail_norm.weight.mul_(scale)
            scaled_modules.append(full_name + ".weight")

            # Do not scale bias.
            # Bias/beta is an additive residual shift and is not part of alpha * F(x).

    if len(scaled_modules) == 0:
        raise RuntimeError(
            "No residual tail normalization layers were found. "
            "Check whether the model uses torchvision-style BasicBlock/Bottleneck "
            "with bn2/bn3 or norm2/norm3."
        )

    if verbose:
        print(
            f"Applied residual scaling: scale={scale}, mode={mode}, "
            f"num_scaled={len(scaled_modules)}"
        )
        for name in scaled_modules:
            print(f"  - {name}")

    return scaled_modules


@torch.no_grad()
def find_sufficiently_small_init(
        model, eval_fn, data, eval_target=1e-3, iterations=8, tolerance=1e-8, scale_function=scaled_model
) -> Tuple[float, float]:
    """
    Find a weights initial scale such that eval_fn() evaluates close to `eval_target`.
    The algo is between hill-climbing and binary search with adjustments to deal better with logarithmic scales.
    """
    x = 1  # x represents the scale factor (independent variable)
    exponent = 0
    target_y = eval_target  # y is whatever we care for (dependent variable)

    last_direction = None
    best_y = None
    best_x = None

    for i in tqdm(range(iterations)):
        # Evaluate current scale factor
        with scale_function(model, x, scale_bias=True) as m:
            y = eval_fn(m, data)

        if i == 0 or abs(math.log10(target_y / y)) < abs(math.log10(target_y / best_y)) or math.isnan(best_y):
            best_x = x
            best_y = y
            worse = False
        else:
            worse = True

        # Decide what scale factor to try next
        dy = target_y - best_y
        if abs(dy) < tolerance:
            return best_x, best_y

        direction = 1 if dy > 0 else -1
        if worse or (last_direction is not None and direction != last_direction):
            # Every time we switch directions (or our step takes us further away from the target value), we decrease
            # the step size
            exponent += 1
        x = best_x * (10 ** (direction * 0.5 ** exponent))
        last_direction = direction
    return best_x, best_y


def dt(data):
    output = []
    for i in range(1, len(data)):
        output.append(data[i] - data[i - 1])
    return output


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def smooth_data(data, window_size=5):
    data_array = np.array(data)
    smooth_data = np.copy(data_array)
    half_window = window_size // 2
    for i in range(len(data_array)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(data_array), i + half_window + 1)
        smooth_data[i] = np.mean(data_array[start_idx:end_idx])
    return smooth_data


@torch.no_grad()
def get_total_weight(model):
    linear_layers = []

    for module in model.modules():
        if isinstance(module, nn.Linear):
            linear_layers.append(module)

    if not linear_layers:
        raise ValueError("no Linear layer")

    total_weight = linear_layers[0].weight

    for layer in linear_layers[1:]:
        total_weight = layer.weight @ total_weight

    return total_weight


def reject_outliers(data, m=0.04):
    indexes = np.where(abs(np.array(dt(data))) > m)
    for i in indexes:
        r = 10
        for j in range(-r, r):
            data[i + j] = data[i + j - 1]
    return data


def write_realtime(path, filename, data):
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/{filename}.csv", 'a+') as f:
        print(f"{data:.8f}", file=f, flush=True)


def write_realtime_seperate(path, filename, datas):
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/{filename}.csv", 'a+') as f:
        print(','.join(f"{data:.8f}" for data in datas), file=f, flush=True)


def write_realtime_list(path, filename, datas):
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/{filename}.csv", 'a+') as f:
        for data in datas:
            print(f"{data:.8f}", file=f, flush=True)


@torch.no_grad()
def compute_matrix_eigenvals(
        model: torch.nn.Module,
        view1, view2,
        representations: bool = False,
        embeddings: bool = False,
        matrix: str = "C",
        use_centering: bool = False,
        use_eval_mode: bool = False,
):
    assert representations ^ embeddings, "Must pick one of representatons or embeddings"
    assert matrix in ("C", "PCA"), f"Expected `matrix` to be either 'C' or 'PCA', got {matrix}"

    original_mode = model.training
    if use_eval_mode:
        model.train(False)

    z1 = model(view1)
    z2 = model(view2)

    batch_size = z1.shape[0]
    if matrix == "C":
        matrix = (z1.T @ z2 + z2.T @ z1) / (2 * batch_size)
    else:  # PCA
        matrix = (z1.T @ z1 + z2.T @ z2) / (2 * batch_size)

    eigenvals = eigenvals_or_nan(matrix)
    eigenvals_np = eigenvals.detach().cpu().numpy()[::-1]

    if use_eval_mode:
        model.train(original_mode)
    return eigenvals_np


def eigenvals_or_nan(x: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) in (2, 3)
    from torch.linalg import LinAlgError
    try:
        return torch.linalg.eigvalsh(x)
    except LinAlgError:
        return torch.full(x.shape[:-1], torch.nan, device=x.device)
