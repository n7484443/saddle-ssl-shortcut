"""Image selection and cached spectral computations for ImageNet."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "feature_preference/practical/imagenet/statics/statics_gamma.py"
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


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
    if not groups:
        raise ValueError("No eligible image groups.")
    return groups


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
