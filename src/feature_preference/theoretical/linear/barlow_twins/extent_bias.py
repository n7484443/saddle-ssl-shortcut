
from pathlib import Path
import sys

src_dir = next(p for p in Path(__file__).resolve().parents
               if (p / "common.py").is_file())
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

    from common import BTLoss, set_seed, scale_weights, get_total_weight
    import yaml
    import matplotlib.colors as mcolors
    import pickle
    import marimo as mo
    import os

    return (
        BTLoss,
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

    abs_config_dir = os.path.abspath(mo.notebook_dir() / "configs")
    with initialize_config_dir(version_base=None, config_dir=abs_config_dir):
        cfg = compose(config_name="config")

    # Example config.yaml:
    #   lambdas: ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"]
    lambdas = [str(l) for l in cfg.lambdas]
    print(lambdas)
    return cfg, lambdas


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
def _(cfg, create_segment_vectors, device, sample, torch):
    # Create lossFn inside train() because it depends on lambda.
    s_in = sum(cfg.sizes)
    s_out = len(cfg.sizes)

    sample_data = sample(cfg.batch, cfg.sizes, device, mul=0.01)
    gamma = torch.bmm(sample_data[0].unsqueeze(dim=2), sample_data[1].unsqueeze(dim=1))  # b x n x n
    gamma = gamma + gamma.permute(0, 2, 1)
    gamma = torch.mean(gamma, dim=0).squeeze() / 2  # n x n

    e = create_segment_vectors(cfg.sizes).to(device)
    return e, gamma, s_in, s_out


@app.cell
def _(gamma, plt, sns, torch):
    sns.heatmap(gamma.cpu())
    plt.show()
    gamma_l, gamma_v = torch.linalg.eig(gamma)
    gamma_l = gamma_l.real.cpu()
    gamma_v = gamma_v.real.cpu()
    print(gamma_l, gamma_v)
    return gamma_l, gamma_v


@app.cell
def _(
    BTLoss,
    cfg,
    device,
    e,
    gamma_v,
    get_total_weight,
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
    def train(lambd, slope=None, verbose=False):
        target_output_dict = {"lambd": str(lambd)}

        # Create the loss for each lambda before set_seed to avoid consuming the seeded RNG.
        lossFn = BTLoss(use_batchnorm=False, in_feature=s_out, lambd=float(lambd)).to(device)

        set_seed(cfg.seed)
        encoder = nn.Linear(s_in, s_out, bias=False, device=device)
        epochs = int(cfg.epochs)
        lr = float(cfg.lr)
        scaling = float(cfg.scaling)

        optimizer = torch.optim.SGD(encoder.parameters(), lr, momentum=0, weight_decay=0)

        encoder.train()

        loss_values = []
        eiglist = [[] for _ in range(s_out)]
        feature_alignment = [[] for _ in range(s_out)]
        weight_singular = [[] for _ in range(s_out)]
        subspace_alignment = []
        feature_weight_r_singular_alignment = [[[] for _ in range(2)] for _ in range(2)]
        scale_weights(encoder, scaling)

        w = get_total_weight(encoder).detach().clone()
        u0, s0, v0T = torch.linalg.svd(w)
        v0 = v0T.T

        if verbose:
            print(encoder)
            print(f"init singular values {s0}")

        target_output_dict["w"] = w.cpu()
        target_output_dict["u0"] = u0.cpu()
        target_output_dict["s0"] = s0.cpu()
        target_output_dict["v0"] = v0.cpu()

        for i in range(s_out):
            weight_singular[i].append(s0[i].cpu().item())
            feature_alignment[i].append(torch.norm(e[i].unsqueeze(dim=0) @ w.T).cpu().item())
        subspace_alignment.append(
            (torch.norm(v0[:, :2].T @ gamma_v[:, :2].cuda(), p="fro") ** 2 / 2).cpu().item())

        for i in range(2):
            for j in range(2):
                feature_weight_r_singular_alignment[i][j].append(
                    ((u0[:, i] @ w @ e[j].cuda()).cpu()).item())

        eig_reversed = False
        bar = tqdm(range(1, epochs + 1), desc=f"lambda={lambd}", leave=False)
        u_prev = None

        for epoch in bar:
            optimizer.zero_grad()
            x0, x1 = sample(cfg.batch, cfg.sizes, device, mul=0.01)
            z0, z1 = torch.chunk(encoder(torch.cat([x0, x1], dim=0)), 2, dim=0)
            c, loss = lossFn(z0, z1)
            c = c.detach()
            v = reversed(torch.linalg.eigvalsh(c))
            for i in range(len(v)):
                eiglist[i].append(v[i].cpu().item())
            loss_values.append(loss.item())
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                bar.set_description(
                    f"lambda={lambd} loss: {loss.item():.4f} {v[0].item():.8f} {v[1].item():.8f}")

            w_current = get_total_weight(encoder).detach().clone()
            u_w_current, s_w_current, v_w_current_h = torch.linalg.svd(w_current)
            v_w_current = v_w_current_h.T
            if epoch > 100:
                s_diff = abs(s_w_current[0] - s_w_current[1])
                if s_diff < 0.0005 and not eig_reversed:
                    eig_reversed = True
                    if verbose:
                        print("reversed")
            if eig_reversed:
                u_w_current = u_w_current[:, [1, 0]]
                s_w_current = s_w_current[[1, 0]]
                v_w_current = v_w_current[:, [1, 0]]
            if u_prev is None:
                u_prev = [u_w_current.clone()]
            if epoch > 100:
                u_prev.pop(0)
                u_prev.append(u_w_current.clone())

            for i in range(s_out):
                weight_singular[i].append(s_w_current[i].cpu().item())
                feature_alignment[i].append(
                    torch.norm(e[i].unsqueeze(dim=0) @ w_current.T).cpu().item())
            subspace_alignment.append(
                (torch.norm(v_w_current[:, :2].T @ gamma_v[:, :2].cuda(), p="fro") ** 2 / 2).cpu().item())

            for i in range(2):
                for j in range(2):
                    feature_weight_r_singular_alignment[i][j].append(
                        ((u_w_current[:, i] @ w_current @ e[j].cuda()).cpu()).item())

        target_output_dict["loss"] = np.array(loss_values)
        target_output_dict["eiglist"] = np.array(eiglist)
        target_output_dict["weight_singular"] = np.array(weight_singular)
        target_output_dict["subspace_alignment"] = np.array(subspace_alignment)
        target_output_dict["feature_weight_r_singular_alignment"] = np.array(
            feature_weight_r_singular_alignment)
        target_output_dict["feature_alignment"] = np.array(feature_alignment)
        return target_output_dict

    def train_all(lambda_list, verbose=False):
        """Run the lambda list and return {lambda(str): output_dict}."""
        results = {}
        for lambd in tqdm(lambda_list, desc="lambdas"):
            results[str(lambd)] = train(lambd, verbose=verbose)
        return results

    return train, train_all


@app.cell
def _(lambdas, train_all):
    output_dict = train_all(lambdas)
    return (output_dict,)


@app.cell
def _(cfg, gamma_l, lambdas, np, os, output_dict, plt, s_out, sns):
    # ---- Loss curves for all lambda values ----
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    epochs_plot = int(cfg.epochs)
    color_continuas = sns.color_palette("rocket", n_colors=len(lambdas))

    for index, lambd_plot in enumerate(lambdas):
        output_target = output_dict[lambd_plot]
        axs.plot(
            output_target["loss"],
            color=color_continuas[index],
            linewidth=5,
            alpha=0.9,
            label=(r"$\alpha = " + f"{lambd_plot}" + "$"),
        )

    # Plot the theoretical prediction once; gamma_l and s0 do not depend on lambda.
    s0_ref = output_dict[lambdas[0]]["s0"]
    t0 = 1450
    t = np.linspace(0, epochs_plot, epochs_plot)
    y1 = gamma_l[0].item() / (
        gamma_l[0].item() + s0_ref[0].item() ** (-2) * np.exp(-8 * gamma_l[0].item() * t / t0))
    y2 = gamma_l[1].item() / (
        gamma_l[1].item() + s0_ref[1].item() ** (-2) * np.exp(-8 * gamma_l[1].item() * t / t0))
    predicted_loss = (y1 - 1) ** 2 + (y2 - 1) ** 2
    axs.plot(predicted_loss, color="blue", linewidth=2.5)

    axs.legend(ncols=2)
    axs.set_xlim(-0.05, min(7000, epochs_plot))
    axs.set_ylim(-0.05, s_out * 1.15)
    axs.set_xlabel("Step")
    axs.set_ylabel("Loss")

    plt.tight_layout()
    os.makedirs("figure/bt", exist_ok=True)
    plt.savefig("figure/bt/loss_only_lambda_all.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(lambdas, output_dict, plt, sns):
    # ---- Correlation matrix eigenvalues for each lambda ----
    fig2, axs2 = plt.subplots(1, 1, figsize=(10, 5))
    color_eig = sns.color_palette("rocket", n_colors=len(lambdas))

    for idx, lambd_eig in enumerate(lambdas):
        eiglist = output_dict[lambd_eig]["eiglist"]
        for k in range(2):
            axs2.plot(
                eiglist[k],
                color=color_eig[idx],
                linewidth=2.5,
                alpha=0.9,
                label=(r"$\alpha = " + f"{lambd_eig}" + "$") if k == 0 else None,
            )

    axs2.set_xlabel("Step")
    axs2.set_ylabel("Eigenvalue")
    axs2.legend(ncols=2)
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()