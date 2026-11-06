import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    import csv
    import hashlib
    import io
    import json
    import math
    import re
    import statistics
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import torch
    import marimo as mo

    device = torch.device("cuda")
    sns.set_theme(style="whitegrid", font="cmr10", font_scale=1.9)
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams["pdf.fonttype"] = 42
    colors = sns.color_palette()
    notebook_dir = Path(__file__).resolve().parent
    workspace = next(p for p in Path(__file__).resolve().parents if (p / "common.py").is_file())
    return (csv, hashlib, io, json, math, mo, notebook_dir, np, plt, re,
            sns, statistics, workspace)


@app.cell
def _(csv, hashlib, math, notebook_dir, re, statistics, workspace):
    # This cell reads existing results only. It never imports training code.
    source_root = workspace / "frequency_shortcut/practical/mnist_backdoor_hadamard_grid"
    selected_rows = []
    source_signatures = []
    checks = []
    for _seed in range(4):
        _folder = source_root / f"result/None-0.7/{_seed}_seed/grid_0.0028_5_0.2subset"
        _file = _folder / "grid_long.csv"
        _config_file = _folder / "config.yaml"
        _config = _config_file.read_text(encoding="utf-8")
        for _setting in (f"seed: {_seed}", "scaling: 0.0028", "subset_ratio: 0.2",
                         "trigger_type: hadamard", "batch_size: 64", "max_epoch: 5"):
            assert _setting in _config, (_config_file, _setting)
        with _file.open(encoding="utf-8", newline="") as _handle:
            _all = list(csv.DictReader(_handle))
        _chosen = [_r for _r in _all if _r["spuriosity"] == "0.7" and int(_r["step"]) == 940]
        assert len(_chosen) == 64, (_seed, len(_chosen))
        _keys = set()
        for _r in _chosen:
            _size, _intensity = int(_r["size_mult"]), float(_r["intensity"])
            _accuracy = float(_r["test_acc"])
            assert int(_r["seed"]) == _seed
            assert int(_r["trigger_len"]) == 16 * _size
            assert int(_r["trigger_rows"]) == math.ceil(16 * _size / 28)
            assert _r["trigger_type"] == "hadamard" and _r["trigger_expand"] == "tile"
            assert float(_r["spuriosity_val"]) == 0.7
            assert float(_r["subset_ratio"]) == 0.2 and float(_r["scaling"]) == 0.0028
            assert int(_r["epoch"]) == 5
            assert math.isfinite(_accuracy) and 0 <= _accuracy <= 100
            assert (_size, _intensity) not in _keys
            _keys.add((_size, _intensity))
            selected_rows.append({"seed": _seed, "size_mult": _size,
                                  "trigger_pixels": 16 * _size, "intensity": _intensity,
                                  "test_accuracy_pct": _accuracy, "step": 940, "spuriosity": 0.7})
        assert _keys == {(s, float(i)) for s in range(1, 9) for i in range(1, 9)}
        checks.append({"seed": _seed, "total_csv_rows": len(_all), "selected_rows": len(_chosen)})
        for _path in (_file, _config_file):
            source_signatures.append({"path": str(_path.relative_to(workspace)),
                                      "sha256": hashlib.sha256(_path.read_bytes()).hexdigest()})
    summary_rows = []
    for _size in range(1, 9):
        for _intensity in range(1, 9):
            _values = [_r["test_accuracy_pct"] for _r in selected_rows
                       if _r["size_mult"] == _size and _r["intensity"] == _intensity]
            assert len(_values) == 4
            summary_rows.append({"size_mult": _size, "trigger_pixels": 16 * _size,
                                 "intensity": _intensity, "mean_accuracy_pct": statistics.mean(_values),
                                 "sample_sd_pp": statistics.stdev(_values), "n_seeds": len(_values)})
    rebuttal_comparison = []
    return checks, rebuttal_comparison, selected_rows, source_signatures, summary_rows


@app.cell
def _(mo):
    mo.md("""
    # Extent and amplitude tradeoff

    Saved MNIST Hadamard-trigger experiments: **seeds 0–3**, **step 940**,
    **spuriosity 0.7**, training subset **20%**. Every heatmap cell uses all four seeds.
    Trigger extent counts overwritten pixels in a 28×28 image; intensity is the
    absolute trigger value after the original MNIST standardization.
    The color scale runs from 70% to 100%; 70% is the expected accuracy of
    relying on the trigger-label association. Values below 70% use the lowest color.

    Cell labels show mean accuracy (%) to one decimal place. The sample SD and full precision values
    are provided separately below and in CSV files. No training runs on redraw.
    """)
    return


@app.cell
def _(np, plt, summary_rows):
    mean_matrix = np.array([r["mean_accuracy_pct"] for r in summary_rows]).reshape(8, 8)
    # Mean accuracy labels only; sample SD remains in the separate table.
    figure, axis = plt.subplots(figsize=(10.6, 8.5))
    heatmap = axis.imshow(mean_matrix, origin="lower", interpolation="nearest",
                          cmap="viridis", vmin=70, vmax=100, aspect="equal",
                          extent=(0.5, 8.5, 0.5, 8.5))
    axis.grid(False)
    for _row in range(8):
        for _column in range(8):
            _value = mean_matrix[_row, _column]
            _rgb = heatmap.cmap(heatmap.norm(_value))[:3]
            _linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
                       for c in _rgb]
            _luminance = sum(c * w for c, w in zip(_linear, (0.2126, 0.7152, 0.0722)))
            _text_color = "black" if _luminance > 0.179 else "white"
            axis.text(_column + 1, _row + 1, f"{_value:.1f}",
                      ha="center", va="center", color=_text_color, fontsize=17)
    axis.set_xticks(range(1, 9))
    axis.set_yticks(range(1, 9), [str(16*i) for i in range(1, 9)])
    axis.set_xlabel("Trigger intensity", labelpad=14)
    axis.set_ylabel("Trigger extent (pixels)", labelpad=17)
    colorbar = figure.colorbar(heatmap, ax=axis, fraction=0.047, pad=0.055,
                              ticks=[70, 75, 80, 85, 90, 95, 100])
    colorbar.set_label("Mean test accuracy (%)", labelpad=16)
    figure.subplots_adjust(left=0.16, bottom=0.15, right=0.84, top=0.97)
    figure
    return figure, mean_matrix


@app.cell
def _(figure, notebook_dir):
    figure.savefig(notebook_dir / "extent_amplitude_heatmap.pdf", bbox_inches="tight", pad_inches=.25)
    figure
    return


if __name__ == "__main__":
    app.run()
