"""Local val adapter and persistent cache for the existing spectral-scaling kernel.

The numerical kernel is loaded from statics_gamma.py, not reimplemented here.
Flat-val pooling and actual XML-class grouping are explicitly distinguished.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "feature_preference/practical/imagenet/statics/statics_gamma.py"
CACHE = ROOT / "cache"
DEFAULTS = dict(area=[0.45, 0.55], grouping="pooled", images_per_group=300,
                k=100, resolution=224, align=True, single_bbox=True,
                min_class_images=2, normalization="group_empirical", seed=0,
                amplitudes=np.linspace(0.2, 2.0, 10).tolist(),
                extents=np.linspace(0.2, 1.0, 9).tolist())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def dataset_identity():
    """Detect added/removed/changed images and XMLs without decoding pixels."""
    identity = hashlib.sha256()
    counts = {}
    for directory, extension in [(ROOT / "val", ".JPEG"), (ROOT / "bbox/val", ".xml")]:
        entries = sorted((p for p in directory.iterdir() if p.suffix == extension), key=lambda p: p.name)
        counts[extension] = len(entries)
        for p in entries:
            stat = p.stat()
            identity.update(f"{p.name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return dict(signature=identity.hexdigest(), counts=counts)


def scan_records(identity):
    key = digest(["legacy-crop-box-scan-v1", identity])
    path = CACHE / "scan" / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["records"], True
    records = []
    from PIL import Image
    for index, xml in enumerate(sorted((ROOT / "bbox/val").glob("*.xml")), 1):
        image = ROOT / "val" / (xml.stem + ".JPEG")
        if not image.exists():
            continue
        tree = ET.parse(xml).getroot()
        # Match the existing notebook's actual JPEG size and bbox rounding.
        with Image.open(image) as im:
            width, height = im.size
        scale = 256 / min(width, height)
        left, top = (round(width * scale) - 224) // 2, (round(height * scale) - 224) // 2
        boxes, classes = [], set()
        for obj in tree.findall("object"):
            classes.add(obj.findtext("name"))
            b = obj.find("bndbox")
            x0, y0, x1, y1 = [int(b.findtext(t)) for t in ("xmin", "ymin", "xmax", "ymax")]
            x0, x1 = [max(0., min(224., x * scale - left)) for x in (x0, x1)]
            y0, y1 = [max(0., min(224., y * scale - top)) for y in (y0, y1)]
            if x1 - x0 >= 4 and y1 - y0 >= 4:
                boxes.append([round(x0), round(y0), round(x1), round(y1)])
        if boxes:
            mask = np.zeros((224, 224), dtype=bool)
            for x0, y0, x1, y1 in boxes:
                mask[y0:y1, x0:x1] = True
            records.append(dict(path=str(image), stem=xml.stem, boxes=boxes,
                                area=float(mask.mean()), class_id=next(iter(classes)) if len(classes) == 1 else None))
        if index % 10000 == 0:
            print(f"Scan: {index:,}/50,000", flush=True)
    write_json(path, dict(identity=identity, records=records))
    return records, False


def select_groups(records, config):
    lo, hi = config["area"]
    selected = [r for r in records if lo <= r["area"] <= hi and
                (not config["single_bbox"] or len(r["boxes"]) == 1)]
    # The original numerical kernel uses boxes[0]; multiple-object input would
    # require a different kernel, so do not silently enable it here.
    if not config["single_bbox"]:
        raise ValueError("The existing kernel supports a single surviving bbox only")
    by_class = {}
    for r in selected:
        if r["class_id"] is not None:
            by_class.setdefault(r["class_id"], []).append(r)
    if config["grouping"] == "pooled":
        groups = {"pooled_val": selected}
    elif config["grouping"] == "classwise":
        groups = {c: rs for c, rs in sorted(by_class.items()) if len(rs) >= config["min_class_images"]}
    else:
        raise ValueError(config["grouping"])
    for name, rows in groups.items():
        limit = config["images_per_group"]
        if limit and len(rows) > limit:
            indices = sorted(np.random.default_rng(config["seed"]).permutation(len(rows))[:limit])
            groups[name] = [rows[int(i)] for i in indices]
    counts = [len(rows) for rows in by_class.values()]
    audit = dict(selected_images=len(selected), selected_classes=len(by_class),
                 max_images_per_class=max(counts, default=0),
                 classes_with_at_least_20=sum(n >= 20 for n in counts),
                 computed_groups=len(groups), computed_images=sum(map(len, groups.values())),
                 group_sizes={name: len(rows) for name, rows in groups.items()})
    if not groups:
        raise ValueError(f"No eligible groups. Selection audit: {audit}")
    return groups, audit


class QuietProgress:
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def update(self, **kwargs): pass


class QuietMarimo:
    status = SimpleNamespace(progress_bar=lambda **kwargs: QuietProgress())
    @staticmethod
    def stop(condition, *args):
        if condition: raise RuntimeError("Kernel stopped")
    @staticmethod
    def md(text): return text


def notebook_cell(marker):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and any(isinstance(child, (ast.FunctionDef, ast.ClassDef)) and child.name == marker
                        for child in n.body))
    node.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["_"], [arg.arg for arg in node.args.args]


def make_kernel(config):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                                    transforms.CenterCrop(224), transforms.ToTensor()])
    value = lambda x: SimpleNamespace(value=x)
    namespace = dict(CROP=224, Dataset=Dataset, F=F, Image=Image, TRANSFORM=transform,
                     torch=torch, ui_align=value(config["align"]), DEVICE=device,
                     DataLoader=DataLoader, math=math, mo=QuietMarimo(), mp=mp, np=np, os=os,
                     run_btn=value(True), ui_bs=value(64), ui_workers=value(0),
                     ui_class_norm=value(True), ui_k=value(config["k"]),
                     ui_res=value(config["resolution"]),
                     ALPHAS=np.asarray(config["amplitudes"]), EXTENTS=np.asarray(config["extents"]))
    namespace["A_REF"] = float(namespace["ALPHAS"][np.argmin(abs(namespace["ALPHAS"] - 1))])
    namespace["E_REF"] = float(namespace["EXTENTS"][-1])
    build, args = notebook_cell("BaseClassDataset")
    namespace["BaseClassDataset"], = build(**{a: namespace[a] for a in args})
    kernel, args = notebook_cell("_condition_eigensystem")
    def compute(name, rows):
        namespace["CLASS_RECORDS"] = {name: rows}
        fa, fe = kernel(**{a: namespace[a] for a in args})
        return dict(f_alpha=fa[name], f_extent=fe[name])
    return compute


def fit_power(x, curves):
    curves = np.asarray(curves)
    if curves.ndim == 1: curves = curves[:, None]
    if not np.isfinite(curves).all() or not (curves > 0).all():
        raise ValueError("Nonpositive or nonfinite spectral strength; do not silently discard modes")
    lx, ly = np.log(np.asarray(x)), np.log(curves)
    slope, intercept = np.polyfit(lx, ly, 1)
    residual = ((ly - (lx[:, None] * slope + intercept)) ** 2).sum(axis=0)
    total = ((ly - ly.mean(axis=0)) ** 2).sum(axis=0)
    r2 = np.divide(residual, total, out=np.full_like(residual, np.nan), where=total > 0)
    return slope, 1 - r2


def summarize(results, config):
    summary = {}
    curves = {}
    for label, key, xs in [("amplitude", "f_alpha", config["amplitudes"]),
                            ("extent", "f_extent", config["extents"])]:
        reference = int(np.argmin(abs(np.asarray(xs) - 1)))
        group_slopes, group_r2, group_mode_medians, all_slopes, all_r2, median_curves = [], [], [], [], [], []
        for result in results.values():
            ratio = result[key] / result[key][reference]
            mode_slopes, mode_r2 = fit_power(xs, ratio)
            median_curve = np.median(ratio, axis=1)
            slope, r2 = fit_power(xs, median_curve)
            median_curves.append(median_curve)
            group_slopes.append(float(slope[0])); group_r2.append(float(r2[0]))
            group_mode_medians.append(float(np.median(mode_slopes)))
            all_slopes.extend(mode_slopes.tolist()); all_r2.extend(mode_r2.tolist())
        summary[label] = dict(
            mean_group_median_curve_exponent=float(np.mean(group_slopes)),
            median_group_curve_r2=float(np.median(group_r2)),
            mean_group_median_mode_exponent=float(np.mean(group_mode_medians)),
            pooled_mode_median_exponent=float(np.median(all_slopes)),
            pooled_mode_median_r2=float(np.median(all_r2)),
            total_modes=len(all_slopes), groups=len(group_slopes))
        curves[label] = dict(x=np.asarray(xs), group_median_curves=np.stack(median_curves, axis=1))
    return summary, curves


def run_cached(config=None):
    config = {**DEFAULTS, **(config or {})}
    if config["normalization"] != "group_empirical":
        raise ValueError("Only fixed empirical group normalization is implemented")
    identity = dataset_identity()
    records, scan_hit = scan_records(identity)
    groups, audit = select_groups(records, config)
    protocol = dict(config=config, dataset=identity,
                    adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    kernel_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    ordering="sorted XML basename; NumPy default_rng(seed) subsample",
                    normalized_space="RGB mean/std fixed per group over all selected pixels",
                    grouping_note="pooled is one mixed-class group, not class-wise normalization")
    key = digest(protocol)
    directory = CACHE / "runs" / key
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "protocol.json", {**protocol, "audit": audit,
               "selected_images": {name: [r["stem"] for r in rows] for name, rows in groups.items()}})
    print(f"Run {key}: {audit}", flush=True) if len(groups) == 1 else print(
        f"Run {key}: {len(groups)} groups, {audit['computed_images']} images", flush=True)
    compute, results, hits = None, {}, 0
    for index, (name, rows) in enumerate(groups.items(), 1):
        path = directory / f"{name}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as stored:
                result = {k: stored[k] for k in stored.files}
            hits += 1
        else:
            if compute is None: compute = make_kernel(config)
            result = compute(name, rows)
            temporary = path.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **result)
            temporary.replace(path)
        results[name] = result
        if index % 25 == 0 or index == len(groups):
            print(f"Groups {index}/{len(groups)}; cache hits {hits}", flush=True)
    summary, curves = summarize(results, config)
    write_json(directory / "summary.json", dict(summary=summary, audit=audit, config=config))
    return dict(key=key, directory=directory, config=config, audit=audit, summary=summary,
                curves=curves, scan_cache_hit=scan_hit, group_cache_hits=hits,
                computed_now=len(groups)-hits)


def plot_result(result):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, label, theory in zip(axes, ("amplitude", "extent"), (2, 1)):
        data = result["curves"][label]
        x, ys = data["x"], data["group_median_curves"]
        curve = np.exp(np.log(ys).mean(axis=1))
        slope, intercept = np.polyfit(np.log(x), np.log(curve), 1)
        ax.loglog(x, curve, "o-", label="Geometric mean of group-median curves")
        ax.loglog(x, np.exp(intercept)*x**slope, "--", label=f"Fit: exponent {slope:.4f}")
        ax.loglog(x, x**theory, ":", color="gray", label=f"Theory: exponent {theory}")
        ax.set(xlabel=label, ylabel="Normalized object-associated strength",
               title=f"{label.capitalize()} · {result['config']['grouping']}")
        ax.grid(True, which="both", alpha=.25); ax.legend(fontsize=8)
    lo, hi = result["config"]["area"]
    fig.suptitle(f"Object ratio {100*lo:g}–{100*hi:g}% · {result['audit']['computed_images']} images · "
                 f"{result['audit']['computed_groups']} groups")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grouping", choices=["pooled", "classwise"], default="pooled")
    parser.add_argument("--images-per-group", type=int, default=300)
    args = parser.parse_args()
    output = run_cached(dict(grouping=args.grouping, images_per_group=args.images_per_group))
    import matplotlib
    matplotlib.use("Agg")
    figure = plot_result(output)
    figure.savefig(output["directory"] / "scaling.png")
    figure.savefig(output["directory"] / "scaling.pdf")
    print(json.dumps(dict(key=output["key"], summary=output["summary"],
                          cache_hits=output["group_cache_hits"], computed_now=output["computed_now"]), indent=2), flush=True)
