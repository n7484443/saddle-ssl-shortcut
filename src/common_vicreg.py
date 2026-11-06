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
    torch.cuda.manual_seed_all(val) # for multi gpu
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
            assert(in_feature > 0)
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
    def __init__(self, batch_size, temperature):
        super().__init__()
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

        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long()

        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return c, loss

class VICRegLoss(nn.Module):
    def __init__(
        self,
        inv_coeff: float = 25.0,
        var_coeff: float = 15.0,
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
        std = x.std(dim=0)
        var_loss = F.relu(gamma - std).mean()
        return var_loss

    @staticmethod
    def covariance_loss(x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=0)
        cov = (x.T @ x) / (x.shape[0] - 1)
        cov_loss = cov.fill_diagonal_(0.0).pow(2).sum() / x.shape[1]
        return cov_loss

@contextmanager
def scaled_model(model, scale, scale_bias):
    assert scale != 0
    try:
        scale_weights(model, scale, scale_bias)
        yield model
    finally:
        scale_weights(model, 1 / scale, scale_bias)

@torch.no_grad()
def scale_weights(model, scale: float = 1, scale_bias=False) -> None:
    for name, param in model.named_parameters():
        if "weight" in name and scale != 1:
            transformed_param = param * scale
            param.copy_(transformed_param)
        elif scale_bias and "bias" in name and scale != 1:
            transformed_param = param * scale
            param.copy_(transformed_param)


@torch.no_grad()
def find_sufficiently_small_init(
        model, eval_fn, data, eval_target=1e-3, iterations=8, tolerance=1e-8
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
        with scaled_model(model, x, scale_bias=True) as m:
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
        output.append(data[i] - data[i-1])
    return output

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
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
        print(f"{data:.6f}", file=f, flush=True)

def write_realtime_seperate(path, filename, datas):
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/{filename}.csv", 'a+') as f:
        print(','.join(f"{data:.6f}" for data in datas), file=f, flush=True)

def write_realtime_list(path, filename, datas):
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/{filename}.csv", 'a+') as f:
        for data in datas:
            print(f"{data:.6f}", file=f, flush=True)