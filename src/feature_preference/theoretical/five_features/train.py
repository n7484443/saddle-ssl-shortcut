from pathlib import Path
import sys

src_dir = next(p for p in Path(__file__).resolve().parents if (p / "common.py").is_file())
sys.path.insert(0, str(src_dir))

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import torch.nn as nn
    from tqdm.auto import tqdm
    import numpy as np
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import seaborn as sns
    import random

    from common import SimCLRLoss, set_seed, scale_weights, get_total_weight
    import yaml
    import matplotlib.colors as mcolors
    import pickle
    import marimo as mo
    import os

    return (
        SimCLRLoss,
        get_total_weight,
        mo,
        nn,
        np,
        os,
        plt,
        scale_weights,
        set_seed,
        sns,
        torch,
        tqdm,
    )


@app.cell
def _(plt, sns, torch):
    device = torch.device("cuda")
    sns.set_theme(style="whitegrid", font="cmr10", font_scale=1.9)
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    colors = sns.color_palette()
    colors
    return (device,)


@app.cell
def _(mo, os):
    from hydra import initialize_config_dir, compose
    abs_config_dir=os.path.abspath(mo.notebook_dir() / "configs")
    with initialize_config_dir(version_base=None, config_dir=abs_config_dir):
        cfg = compose(config_name="config")
    return (cfg,)


@app.cell
def _(torch):
    def sample(batch, sizes, device, mul=0.01):
        x_out = torch.zeros((batch, 0), device=device)
        for size in sizes:
            rand = torch.rand((batch, 1), device=device).round() * 2 - 1
            x_out = torch.cat([x_out, rand.repeat(1, size)], dim=-1)
        epsilon1 = mul * torch.randn_like(x_out)
        epsilon2 = mul * torch.randn_like(x_out)
        return x_out + epsilon1, x_out + epsilon2

    def create_segment_vectors(sizes):
        total_length = sum(sizes)
        cumsum = torch.cumsum(torch.tensor(sizes), dim=-1)

        result = torch.zeros((len(sizes), total_length))

        for i in range(len(sizes)):
            start_idx = cumsum[i - 1] if i > 0 else 0
            end_idx = cumsum[i]

            result[i, start_idx:end_idx] = 1

        return result

    return create_segment_vectors, sample


@app.cell
def _(SimCLRLoss, cfg, create_segment_vectors, device, sample, torch):
    s_in = sum(cfg.sizes)
    s_out = len(cfg.sizes)

    sample_data = sample(cfg.batch, cfg.sizes, device, mul=0.01)
    gamma = torch.bmm(sample_data[0].unsqueeze(dim=2), sample_data[1].unsqueeze(dim=1))  # b x n x n
    gamma = gamma + gamma.permute(0, 2, 1)
    gamma = torch.mean(gamma, dim=0).squeeze() / 2  # n x n

    e = create_segment_vectors(cfg.sizes).to(device)

    lossFn = SimCLRLoss(cfg.batch, 0.5).to(device)
    return e, gamma, lossFn, s_in, s_out


@app.cell
def _(gamma, plt, sns, torch):
    sns.heatmap(gamma.cpu())
    plt.show()
    gamma_l, gamma_v = torch.linalg.eig(gamma)
    gamma_l = gamma_l.real.cpu()
    gamma_v = gamma_v.real.cpu()
    print(gamma_l, gamma_v)
    return (gamma_v,)


@app.cell
def _(
    cfg,
    device,
    e,
    gamma_v,
    get_total_weight,
    lossFn,
    nn,
    np,
    s_in,
    s_out,
    sample,
    scale_weights,
    set_seed,
    torch,
    tqdm,
):
    def train(lambd=None, slope=None):
        target_output_dict = {}
        set_seed(cfg.seed)
        encoder = nn.Linear(s_in, s_out, bias=False, device=device)
        print(encoder)
        epochs = int(cfg.epochs)
        lr = float(cfg.lr)
        scaling = float(cfg.scaling)

        optimizer = torch.optim.SGD(encoder.parameters(), lr, momentum=0, weight_decay=0)

        encoder.train()

        loss_values = []
        eiglist = [[] for i in range(s_out)]
        feature_alignment = [[] for _ in range(s_out)]
        weight_singular = [[] for _ in range(s_out)]
        subspace_alignment = []
        feature_weight_r_singular_alignment = [[[] for _ in range(2)] for _ in range(2)]
        scale_weights(encoder, scaling)

        w = get_total_weight(encoder).detach().clone()
        u0, s0, v0T = torch.linalg.svd(w)
        v0 = v0T.T

        print(f"before {s0}")
        w = get_total_weight(encoder).detach().clone()
        u0, s0, v0T = torch.linalg.svd(w)
        print(f"after {s0}")
        v0 = v0T.T

        target_output_dict["w"] = w.cpu()
        target_output_dict["u0"] = u0.cpu()
        target_output_dict["s0"] = s0.cpu()
        target_output_dict["v0"] = v0.cpu()

        for i in range(s_out):
            weight_singular[i].append(s0[i].cpu().item())
            feature_alignment[i].append(torch.norm(e[i].unsqueeze(dim=0) @ w.T).cpu().item())
        subspace_alignment.append(
            (torch.norm(v0[:, :2].T @ gamma_v[:, :2].cuda(), p='fro') ** 2 / 2).cpu().item())
        print(subspace_alignment[0])

        for i in range(2):
            for j in range(2):
                feature_weight_r_singular_alignment[i][j].append(
                    ((u0[:, i] @ w @ e[j].cuda()).cpu()).item())

        eig_reversed = False
        bar = tqdm(range(1, cfg.epochs + 1))
        u_prev = None

        for epoch in bar:
            optimizer.zero_grad()
            x0, x1 = sample(cfg.batch, cfg.sizes, device, mul=0.01)
            z0, z1 = torch.chunk(encoder(torch.cat([x0, x1], dim=0)), 2, dim=0)
            c, loss = lossFn(z0, z1)
            c = c.detach()
            v = reversed(torch.linalg.eigvalsh(c))
            for i in range(len(v)):
                # print(s0[0].item()**2 * gamma_l[0].item(), v[0].item())
                eiglist[i].append(v[i].cpu().item())
            loss_values.append(loss.item())
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                bar.set_description(f'loss: {loss.item():.4f} {v[0].item():.8f}  {v[1].item():.8f}')

            w_current = get_total_weight(encoder).detach().clone()
            u_w_current, s_w_current, v_w_current_h = torch.linalg.svd(w_current)
            v_w_current = v_w_current_h.T
            if epoch > 100:
                s_diff = abs(s_w_current[0] - s_w_current[1])
                if s_diff < 0.0005 and not eig_reversed:
                    eig_reversed = True
                    print("reversed")
            if eig_reversed:
                # Swap the first two tracked modes without dropping the others.
                mode_order = list(range(s_out))
                mode_order[0], mode_order[1] = mode_order[1], mode_order[0]
                u_w_current = u_w_current[:, mode_order]
                s_w_current = s_w_current[mode_order]
                v_w_current = v_w_current[:, mode_order]
            if u_prev is None:
                u_prev = [u_w_current.clone()]
            if epoch > 100:
                u_prev.pop(0)
                u_prev.append(u_w_current.clone())

            for i in range(s_out):
                weight_singular[i].append(s_w_current[i].cpu().item())
                feature_alignment[i].append(torch.norm(e[i].unsqueeze(dim=0) @ w_current.T).cpu().item())
            subspace_alignment.append(
                (torch.norm(v_w_current[:, :2].T @ gamma_v[:, :2].cuda(), p='fro') ** 2 / 2).cpu().item())

            for i in range(2):
                for j in range(2):
                    feature_weight_r_singular_alignment[i][j].append(
                        ((u_w_current[:, i] @ w_current @ e[j].cuda()).cpu()).item())

        feature_weight_r_singular_alignment = np.array(feature_weight_r_singular_alignment)
        target_output_dict["loss"] = np.array(loss_values)
        target_output_dict["eiglist"] = np.array(eiglist)
        target_output_dict["weight_singular"] = np.array(weight_singular)
        target_output_dict["subspace_alignment"] = np.array(subspace_alignment)
        target_output_dict["feature_weight_r_singular_alignment"] = feature_weight_r_singular_alignment
        target_output_dict["feature_alignment"] = np.array(feature_alignment)
        return target_output_dict


    return (train,)


@app.cell
def _(plt, train):
    target_output_dict = train()
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    _axes[0].plot(target_output_dict["loss"])
    _axes[0].set(xlabel="Training step (zero-based)", ylabel="SimCLR loss")
    for _i, _curve in enumerate(target_output_dict["eiglist"]):
        _axes[1].plot(_curve, label=f"Mode {_i + 1}")
    _axes[1].set(xlabel="Training step (zero-based)", ylabel="Cross-covariance eigenvalue")
    _axes[1].legend()
    _fig.tight_layout()
    _fig
    return (target_output_dict,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
