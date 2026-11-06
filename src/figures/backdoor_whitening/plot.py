import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import csv
    import hashlib
    import io
    import json
    import math
    import statistics
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch

    device = torch.device("cuda")
    sns.set_theme(style="whitegrid", font="cmr10", font_scale=1.9)
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'pdf.use14corefonts': False})
    colors = sns.color_palette()
    return Path, colors, csv, hashlib, io, json, math, mo, plt, statistics


@app.cell
def _(Path, csv, hashlib, json, math, statistics):
    output_dir = Path(__file__).resolve().parent
    project_dir = next(p for p in Path(__file__).resolve().parents if (p / "common.py").is_file())
    practical_dir = project_dir / "frequency_shortcut/practical"
    source_dirs = {
        "Without whitening": practical_dir / "mnist_backdoor/result/None",
        "With ZCA whitening": practical_dir / "mnist_backdoor_whitening/result/None",
    }
    seeds = tuple(range(6))
    expected_intensities = [0.1, 0.154, 0.2371, 0.3652, 0.5623, 0.866,
        1.3335, 2.0535, 3.1623, 4.8697, 7.4989, 11.5478, 17.7828,
        27.3842, 42.1697, 74.989, 100.0]
    records = []
    source_signatures = []
    for _condition, _source_dir in source_dirs.items():
        for _seed in seeds:
            _path = _source_dir / f"{_seed}_seed/0.0028_5step_0.2subset.csv"
            with _path.open(newline="", encoding="utf-8-sig") as _f:
                _rows = list(csv.DictReader(_f))
            _seen = set()
            for _row in _rows:
                _step = int(_row["step"])
                _intensity = float(_row["intensity"])
                _accuracy = float(_row["test_acc"])
                if not (math.isfinite(_intensity) and _intensity > 0):
                    raise ValueError(f"Nonpositive/nonfinite intensity: {_path}")
                if not (math.isfinite(_accuracy) and 0 <= _accuracy <= 100):
                    raise ValueError(f"Invalid accuracy: {_path}")
                if (_step, _intensity) in _seen:
                    raise ValueError(f"Duplicate step/intensity: {_path}")
                _seen.add((_step, _intensity))
                if _step == 940:
                    records.append(dict(condition=_condition, seed=_seed, step=_step,
                        intensity=_intensity, test_accuracy_percent=_accuracy))
            _selected = sorted(_i for _s, _i in _seen if _s == 940)
            if _selected != expected_intensities:
                raise ValueError(f"Incomplete or unexpected intensity grid: {_path}")
            source_signatures.append(dict(path=str(_path.relative_to(project_dir)),
                sha256=hashlib.sha256(_path.read_bytes()).hexdigest(),
                rows=len(_rows), all_steps=sorted({_s for _s, _i in _seen})))
    if len(records) != 2 * 6 * 17:
        raise ValueError("Expected exactly 204 selected observations")
    summary = []
    for _condition in source_dirs:
        for _intensity in expected_intensities:
            _values = [_r["test_accuracy_percent"] for _r in records
                if _r["condition"] == _condition and _r["intensity"] == _intensity]
            summary.append(dict(condition=_condition, intensity=_intensity,
                mean_accuracy_percent=statistics.mean(_values),
                sample_sd_pp=statistics.stdev(_values), n_seeds=len(_values)))
    historical_rows = []
    _published = {
        "Without whitening": [88.05, 88.86, 87.98, 11.31, 10.74],
        "With ZCA whitening": [84.06, 84.58, 84.47, 85.20, 84.75],
    }
    for _condition, _published_values in _published.items():
        for _intensity, _reference in zip([0.1, 0.5623, 3.1623, 17.7828, 100.0], _published_values):
            _values = [_r["test_accuracy_percent"] for _r in records
                if _r["condition"] == _condition and _r["intensity"] == _intensity
                and _r["seed"] in (0, 1)]
            _mean = statistics.mean(_values)
            historical_rows.append(dict(condition=_condition, intensity=_intensity,
                seeds_0_1_mean=_mean, rebuttal_accuracy=_reference,
                difference_pp=_mean-_reference))
    for _filename, _data in [("summary.csv", summary),
                            ("selected_observations.csv", records),
                            ("historical_two_seed_audit.csv", historical_rows)]:
        with (output_dir / _filename).open("w", newline="", encoding="utf-8") as _f:
            _writer = csv.DictWriter(_f, fieldnames=list(_data[0]))
            _writer.writeheader()
            _writer.writerows(_data)
    implementation_sources = []
    for _rel in ["mnist_backdoor_whitening/mnist_backdoor.py",
                 "mnist_backdoor_whitening/configs/conf.yaml",
                 
                 "mnist_backdoor/mnist_backdoor.py",
                 ]:
        _path = practical_dir / _rel
        implementation_sources.append(dict(path=str(_path.relative_to(project_dir)),
            sha256=hashlib.sha256(_path.read_bytes()).hexdigest()))
    audit = dict(seeds=list(seeds), step=940, intensity_count=17,
        zero_intensity_present=False, selected_observations=len(records),
        statistic="arithmetic mean and sample SD (ddof=1), across seeds",
        sources=source_signatures, implementation_sources=implementation_sources,
        historical_two_seed_comparison=historical_rows,
        zca_audit=dict(epsilon_in_current_config=0.5, function_fallback_epsilon=0.01,
            fitted_on="12,000 training images (20% subset), after normalization and trigger insertion, before augmentation",
            reused_for="nonaugmented linear-probe train and full test set",
            caution="CSV files do not contain frozen run configurations; epsilon is supported by current code/config, not per-run metadata."))
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit, historical_rows, output_dir, records, summary


@app.cell
def _(mo):
    mo.md("""
    # Whitening and trigger intensity

    Saved step **940**, random trigger-label association (**10% expected agreement**).
    All six common seeds (0-5) and all 17 positive intensities are included.
    Error bars show **sample standard deviation across seeds**, not confidence intervals.
    This notebook only reads saved results; it never launches training.
    """)
    return


@app.cell
def _(colors, io, mo, output_dir, plt, summary):
    def plot_whitening(rows):
        fig, ax = plt.subplots(figsize=(12.2, 6.9))
        for condition, color, marker in [
            ("Without whitening", colors[0], "o"),
            ("With ZCA whitening", colors[1], "s"),
        ]:
            selected = [r for r in rows if r["condition"] == condition]
            ax.errorbar([r["intensity"] for r in selected],
                [r["mean_accuracy_percent"] for r in selected],
                yerr=[r["sample_sd_pp"] for r in selected],
                color=color, marker=marker, linewidth=2.5, markersize=6,
                elinewidth=1.5, capsize=3.5, label=condition)
        ax.axhline(10, color="0.45", linewidth=1.4, linestyle="--", zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(0.08, 125)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Trigger intensity")
        ax.set_ylabel("Linear-probe test accuracy (%)")
        ax.set_yticks([0, 10, 30, 50, 70, 90, 100])
        ax.legend(loc="center left", frameon=True)
        ax.grid(which="major", alpha=0.5)
        fig.tight_layout(pad=1.4)
        return fig

    whitening_figure = plot_whitening(summary)
    whitening_figure.savefig(output_dir / "whitening_comparison.png",
        bbox_inches="tight", pad_inches=0.18)
    whitening_figure.savefig(output_dir / "whitening_comparison.pdf",
        bbox_inches="tight", pad_inches=0.18)
    _buffer = io.BytesIO()
    whitening_figure.savefig(_buffer, format="pdf", bbox_inches="tight", pad_inches=0.18)
    pdf_download = mo.download(data=_buffer.getvalue(), filename="whitening_comparison.pdf",
        mimetype="application/pdf", label="Download PDF")
    mo.vstack([whitening_figure, pdf_download])
    return plot_whitening, whitening_figure


@app.cell
def _(historical_rows, mo, summary):
    mo.accordion({
        "Numerical summary (separate from figure)": mo.ui.table(summary),
        "Historical two-seed audit": mo.vstack([
            mo.md("The historical table approximately matches seeds 0 and 1. At intensity 0.1, whitening gives 84.045%, versus the reported 84.06% (difference -0.015 percentage points). This audit does not select seeds for the new six-seed figure."),
            mo.ui.table(historical_rows),
        ]),
        "ZCA implementation audit": mo.md("Current config uses regularized ZCA with epsilon 0.5. The transform is fitted on the 12,000-image training subset after normalization and trigger insertion, before augmentation, and reused for probe/test images. Run-specific frozen configurations are absent from these CSVs. Regularized ZCA does not make every covariance eigenvalue exactly one and does not establish identity cross-view covariance."),
    })
    return


if __name__ == "__main__":
    app.run()
