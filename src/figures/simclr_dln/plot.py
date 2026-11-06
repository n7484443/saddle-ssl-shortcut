import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import os
    import sys
    HERE = Path(__file__).resolve().parent
    ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common.py").is_file())
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(HERE / "mpl_config"))
    import hashlib
    import json
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns
    import marimo as mo
    device = torch.device("cuda")
    sns.set_theme(style="whitegrid", font="cmr10", font_scale=1.9)
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams.update({"mathtext.fontset": "cm", "pdf.fonttype": 42, "ps.fonttype": 42, "pdf.use14corefonts": False})
    colors = sns.color_palette()
    return HERE, ROOT, colors, hashlib, json, mo, np, os, plt


@app.cell
def _(ROOT, hashlib, json, np):
    run_dirs = {
        "DLN": HERE / "dln",
        "MLP": HERE / "mlp",
    }
    losses, metadata, cache_hashes = {}, {}, {}
    for _name, _folder in run_dirs.items():
        _cache = _folder / "traces.npz"
        cache_hashes[_name] = hashlib.sha256(_cache.read_bytes()).hexdigest()
        with np.load(_cache, allow_pickle=False) as _z:
            losses[_name] = _z["loss"].copy()
        metadata[_name] = json.loads((_folder / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["DLN"]["activation"] == "none"
    assert metadata["DLN"]["network"] == "deep"
    return cache_hashes, losses, metadata, run_dirs


@app.cell
def _(mo):
    mo.md("""# Figure 8(c): linear alternative

    Cached four-layer DLN + SimCLR loss; no training is performed here.
    The comparison uses existing runs with different initialization scales
    (DLN 4e-9; ReLU MLP 4e-8), so transition-time differences are not an
    activation-only controlled comparison. These old runs constructed Linear
    parameters on CUDA; this redraw is not a new CPU-initialized experiment.
    """)
    return


@app.cell
def _(HERE, ROOT, colors, losses, np, os, plt):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    dln_line, = ax.plot(np.arange(len(losses["DLN"])), losses["DLN"], color=colors[2], linewidth=2, alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    fig.tight_layout()
    output_dir = HERE
    if os.environ.get("FIG8C_EXPORT") == "1":
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "dim_loss.pdf", bbox_inches="tight", pad_inches=0.12)
        fig.savefig(output_dir / "dim_loss.png", bbox_inches="tight", pad_inches=0.12)
    assert np.array_equal(dln_line.get_ydata(), losses["DLN"])
    fig
    return dln_line, fig, output_dir


@app.cell
def _(colors, losses, np, os, output_dir, plt):
    comparison, axs = plt.subplots(1, 2, figsize=(12.8, 4.4))
    for _ax, _name, _title in zip(axs, ["DLN", "MLP"], ["DLN", "MLP (ReLU)"]):
        _line, = _ax.plot(np.arange(len(losses[_name])), losses[_name], color=colors[2], linewidth=2, alpha=0.7)
        _ax.set_xlabel("Step")
        _ax.set_ylabel("Loss")
        _ax.set_title(_title)
        assert np.array_equal(_line.get_ydata(), losses[_name])
    comparison.tight_layout()
    if os.environ.get("FIG8C_EXPORT") == "1":
        comparison.savefig(output_dir / "comparison.pdf", bbox_inches="tight", pad_inches=0.12)
        comparison.savefig(output_dir / "comparison.png", bbox_inches="tight", pad_inches=0.12)
    comparison
    return comparison,


@app.cell
def _(HERE, cache_hashes, hashlib, json, losses, metadata, np, os, output_dir, run_dirs):
    if os.environ.get("FIG8C_EXPORT") == "1":
        _unchanged = all(hashlib.sha256((run_dirs[_name] / "traces.npz").read_bytes()).hexdigest() == _sha
                         for _name, _sha in cache_hashes.items())
        assert _unchanged
        _summary = {
            "no_new_training": True, "cache_hashes": cache_hashes, "cache_unchanged": _unchanged,
            "output_directory": str(output_dir), "raw_trace_equal_to_plotted_y": True,
            "dpi_specified": False, "tick_overrides": False,
            "initialization_caveat": "Existing runs use Linear(device=cuda); no CPU-initialized rerun in this task.",
            "runs": {name: {"run_directory": str(run_dirs[name]), "network": metadata[name]["network"],
                            "activation": metadata[name]["activation"], "steps": len(losses[name]),
                            "batch": metadata[name]["batch_size"], "lr": metadata[name]["learning_rate"],
                            "scaling": metadata[name]["scaling"], "seed": metadata[name]["config"]["SimCLR"]["seed"],
                            "loss_start": float(losses[name][0]), "loss_end": float(losses[name][-1]),
                            "sampled_losses": {str(i): float(losses[name][i]) for i in [100, 200, 400, 500, 550, 600, 800]}}
                     for name in losses},
        }
        (HERE / "verification.json").write_text(json.dumps(_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return


if __name__ == "__main__":
    app.run()
