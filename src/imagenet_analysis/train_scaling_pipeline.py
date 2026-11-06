"""Unattended train-subset preparation and bounded, source-based scaling study.

Waits for the independently verified train download. Never unpacks the full
training dataset. Keeps all outcomes rather than selecting for rebuttal numbers.
"""
from __future__ import annotations

import ast
import ctypes
import hashlib
import json
import math
import multiprocessing as mp
import msvcrt
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import time
import xml.etree.ElementTree as ET

import numpy as np
import scaling_cache as sc

ROOT = Path(__file__).resolve().parent
STATUS = ROOT / "train_scaling_status.json"
BOX_ARCHIVE = ROOT / "bbox/ILSVRC2012_bbox_train_v2.tar.gz"
TRAIN_ARCHIVE = ROOT / "ILSVRC2012_img_train.tar"
CATALOG = ROOT / "cache/train_bbox_catalog.json"
RECORDS = ROOT / "cache/train_records.json"
STATE = dict(pid=os.getpid(), status="starting")
STUDY = [
    dict(name="train_45_55_classnorm", area=[.45, .55], normalization="class_empirical"),
    dict(name="train_40_60_classnorm", area=[.4, .6], normalization="class_empirical"),
    dict(name="train_45_55_globalnorm", area=[.45, .55], normalization="global_empirical"),
    dict(name="train_40_60_globalnorm", area=[.4, .6], normalization="global_empirical"),
]


def status(**updates):
    STATE.update(updates, updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    sc.write_json(STATUS, STATE)


def catalog():
    checksum = hashlib.md5(BOX_ARCHIVE.read_bytes()).hexdigest()
    if checksum != "9271167e2176350e65cfe4e546f14b17":
        raise ValueError("Train bbox checksum mismatch")
    if CATALOG.exists():
        saved = json.loads(CATALOG.read_text(encoding="utf-8"))
        if saved["bbox_md5"] == checksum and saved["class_limit"] == 200:
            return saved
    status(status="cataloging_bbox")
    class_counts = {}
    with tarfile.open(BOX_ARCHIVE, "r|gz") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".xml"):
                cls = PurePosixPath(member.name).parts[-2]
                if not re.fullmatch(r"n\d{8}", cls):
                    raise ValueError(member.name)
                class_counts[cls] = class_counts.get(cls, 0) + 1
    classes = sorted(class_counts)[:200]
    keep = set(classes)
    rows = []
    with tarfile.open(BOX_ARCHIVE, "r|gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".xml"):
                continue
            path = PurePosixPath(member.name)
            cls, stem = path.parts[-2], path.stem
            if cls not in keep:
                continue
            if not re.fullmatch(r"n\d{8}_\d+", stem):
                raise ValueError(stem)
            xml = ET.fromstring(tar.extractfile(member).read())
            boxes = []
            for obj in xml.findall("object"):
                b = obj.find("bndbox")
                boxes.append([int(b.findtext(t)) for t in ("xmin", "ymin", "xmax", "ymax")])
            rows.append(dict(class_id=cls, stem=stem, raw_boxes=boxes,
                             xml_width=int(xml.findtext("size/width")),
                             xml_height=int(xml.findtext("size/height"))))
    rows.sort(key=lambda r: (r["class_id"], r["stem"]))
    saved = dict(bbox_md5=checksum, class_limit=200, classes=classes,
                 total_annotation_classes=len(class_counts), total_annotations=sum(class_counts.values()),
                 selected_annotation_count=len(rows), records=rows)
    sc.write_json(CATALOG, saved)
    print(f"BBox catalog ready: first 200 sorted classes, {len(rows):,} annotated images.", flush=True)
    return saved


def wait_for_download():
    status(status="waiting_for_verified_download")
    while True:
        p = ROOT / "train_download_status.json"
        current = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        phase = current.get("status", "not_started")
        status(download_status=phase, download_percent=current.get("percent"))
        if phase == "complete":
            if current.get("actual_md5") != "1d675b47d978889d74fa0da5fadfb00e":
                raise ValueError("Download completion lacks the expected MD5 verification")
            if TRAIN_ARCHIVE.stat().st_size != 147897477120:
                raise ValueError("Train archive size mismatch")
            return
        if phase == "failed":
            print("Downloader reports failure; waiting for its monitored recovery.", flush=True)
        time.sleep(30)


def extract_annotated_subset(cat):
    wanted = {}
    for row in cat["records"]:
        wanted.setdefault(row["class_id"], set()).add(row["stem"] + ".JPEG")
    root = ROOT / "train"
    root.mkdir(exist_ok=True)
    marker = ROOT / "cache/train_extraction.json"
    signature = sc.digest(dict(classes=cat["classes"], bbox=cat["bbox_md5"], archive_md5="1d675b47d978889d74fa0da5fadfb00e"))
    if marker.exists():
        old = json.loads(marker.read_text(encoding="utf-8"))
        if old.get("signature") == signature and all((root / cls / name).exists() for cls, names in wanted.items() for name in names):
            print("Annotated train subset already extracted.", flush=True)
            return
    status(status="extracting_annotated_train_subset")
    missing, count = [], 0
    with tarfile.open(TRAIN_ARCHIVE, "r:") as outer:
        outer_members = {PurePosixPath(m.name).stem: m for m in outer if m.isfile()}
        for ci, cls in enumerate(cat["classes"], 1):
            if cls not in outer_members:
                raise ValueError(f"Class archive missing: {cls}")
            directory = root / cls
            directory.mkdir(exist_ok=True)
            found = set()
            with outer.extractfile(outer_members[cls]) as class_stream, tarfile.open(fileobj=class_stream, mode="r:") as inner:
                for member in inner:
                    if not member.isfile():
                        continue
                    name = PurePosixPath(member.name).name
                    if name not in wanted[cls]:
                        continue
                    found.add(name)
                    target = directory / name
                    if not target.exists():
                        if shutil.disk_usage(ROOT).free < member.size + 10 * 1024**3:
                            raise OSError("Less than 10 GiB free space remains; stopping safe subset extraction")
                        temporary = target.with_suffix(".JPEG.part")
                        with inner.extractfile(member) as src, temporary.open("wb") as dst:
                            shutil.copyfileobj(src, dst, 1024 * 1024)
                        temporary.rename(target)
                    if target.stat().st_size != member.size:
                        raise ValueError(f"Extracted image size mismatch: {target}")
                    count += 1
            missing.extend(f"{cls}/{name}" for name in sorted(wanted[cls] - found))
            if ci % 10 == 0:
                print(f"Extracted {ci}/200 classes, {count:,} annotated images.", flush=True)
                status(extracted_classes=ci, extracted_images=count)
    sc.write_json(marker, dict(signature=signature, extracted_images=count, missing_images=missing))
    if missing:
        raise ValueError(f"{len(missing)} bbox image IDs were not in the train archive; inspect extraction manifest")


def scan_train(cat):
    fingerprint = hashlib.sha256()
    for row in cat["records"]:
        path = ROOT / "train" / row["class_id"] / (row["stem"] + ".JPEG")
        stat = path.stat()
        fingerprint.update(f"{row['stem']}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    identity = dict(bbox_md5=cat["bbox_md5"], image_metadata_sha256=fingerprint.hexdigest(),
                    classes=cat["classes"], crop_protocol="legacy-statics-gamma-224-v1")
    if RECORDS.exists():
        old = json.loads(RECORDS.read_text(encoding="utf-8"))
        if old["identity"] == identity:
            return old
    from PIL import Image
    status(status="scanning_train_geometry")
    records, errors, mismatched_sizes = [], [], []
    for i, row in enumerate(cat["records"], 1):
        path = ROOT / "train" / row["class_id"] / (row["stem"] + ".JPEG")
        try:
            with Image.open(path) as image:
                w, h = image.size
            if (w, h) != (row["xml_width"], row["xml_height"]):
                mismatched_sizes.append(row["stem"])
            scale = 256 / min(w, h)
            left, top = (round(w*scale)-224)//2, (round(h*scale)-224)//2
            boxes = []
            for x0, y0, x1, y1 in row["raw_boxes"]:
                x0, x1 = [max(0., min(224., x*scale-left)) for x in (x0, x1)]
                y0, y1 = [max(0., min(224., y*scale-top)) for y in (y0, y1)]
                if x1-x0 >= 4 and y1-y0 >= 4:
                    boxes.append([round(x0), round(y0), round(x1), round(y1)])
            if boxes:
                mask = np.zeros((224, 224), dtype=bool)
                for x0, y0, x1, y1 in boxes:
                    mask[y0:y1, x0:x1] = True
                records.append(dict(path=str(path), stem=row["stem"], class_id=row["class_id"],
                                    boxes=boxes, area=float(mask.mean())))
        except Exception as exc:
            errors.append(dict(stem=row["stem"], error=str(exc)))
        if i % 10000 == 0:
            print(f"Train geometry {i:,}/{len(cat['records']):,}", flush=True)
            status(scanned_images=i, total_images=len(cat["records"]))
    result = dict(identity=identity, records=records, image_read_errors=errors,
                  xml_image_dimension_mismatches=mismatched_sizes)
    sc.write_json(RECORDS, result)
    return result


def global_stats(groups, data_identity):
    import torch
    from torchvision import transforms
    from PIL import Image
    key = sc.digest([data_identity, {k: [r["stem"] for r in rs] for k, rs in groups.items()}, "global-rgb-stats-v1"])
    path = ROOT / "cache/train_global_stats" / f"{key}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        return torch.tensor(saved["mean"]).view(3, 1, 1), torch.tensor(saved["std"]).view(3, 1, 1)
    transform = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                                    transforms.CenterCrop(224), transforms.ToTensor()])
    sums, squares = torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    pixels = 0
    for ci, rows in enumerate(groups.values(), 1):
        for start in range(0, len(rows), 64):
            images = []
            for row in rows[start:start+64]:
                with Image.open(row["path"]) as im:
                    images.append(transform(im.convert("RGB")))
            batch = torch.stack(images).double()
            sums += batch.sum((0, 2, 3)); squares += (batch*batch).sum((0, 2, 3))
            pixels += batch.shape[0]*224*224
        if ci % 20 == 0: status(normalization_classes=ci)
    mean = sums / pixels
    std = (squares / pixels - mean.square()).clamp_min(1e-12).sqrt()
    mean, std = mean.float().view(3, 1, 1), std.float().view(3, 1, 1)
    sc.write_json(path, dict(mean=mean.flatten().tolist(), std=std.flatten().tolist(), pixels=pixels))
    return mean, std


def global_kernel(config, fixed_stats):
    """Reuse the original cell, supplying its precomputed fixed RGB statistics.

    Only the stats-provider body is replaced; all spectral math is unchanged.
    Running one class at a time enables per-class checkpoint/cache reuse.
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    value = lambda x: type("Value", (), {"value": x})()
    transform = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                                    transforms.CenterCrop(224), transforms.ToTensor()])
    namespace = dict(CROP=224, Dataset=Dataset, F=F, Image=Image, TRANSFORM=transform, torch=torch,
                     ui_align=value(config["align"]), DEVICE=torch.device("cuda"), DataLoader=DataLoader,
                     math=math, mo=sc.QuietMarimo(), mp=mp, np=np, os=os, run_btn=value(True),
                     ui_bs=value(64), ui_workers=value(0), ui_class_norm=value(True),
                     ui_k=value(config["k"]), ui_res=value(config["resolution"]),
                     ALPHAS=np.asarray(config["amplitudes"]), EXTENTS=np.asarray(config["extents"]),
                     A_REF=1.0, E_REF=1.0)
    build, args = sc.notebook_cell("BaseClassDataset")
    namespace["BaseClassDataset"], = build(**{a: namespace[a] for a in args})
    tree = ast.parse(sc.SOURCE.read_text(encoding="utf-8"))
    cell = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and any(
        isinstance(child, ast.FunctionDef) and child.name == "_condition_eigensystem" for child in n.body))
    provider = next(n for n in cell.body if isinstance(n, ast.FunctionDef) and n.name == "_class_empirical_stats")
    provider.body = ast.parse("return _fixed_mean, _fixed_std").body
    cell.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[cell], type_ignores=[]))
    scope = dict(_fixed_mean=fixed_stats[0], _fixed_std=fixed_stats[1])
    exec(compile(module, str(sc.SOURCE), "exec"), scope)
    def compute(name, rows):
        namespace["CLASS_RECORDS"] = {name: rows}
        fa, fe = scope["_"](**{a.arg: namespace[a.arg] for a in cell.args.args})
        return dict(f_alpha=fa[name], f_extent=fe[name])
    return compute


def compute_run(config, data, name):
    groups, audit = sc.select_groups(data["records"], config)
    protocol = dict(config=config, dataset=data["identity"],
                    original_kernel_sha256=hashlib.sha256(sc.SOURCE.read_bytes()).hexdigest(),
                    pipeline_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    adapter_sha256=hashlib.sha256(Path(sc.__file__).read_bytes()).hexdigest(),
                    class_selection="first 200 sorted bbox classes before area filtering",
                    image_order="sorted filename before deterministic seed-0 subsampling",
                    audit=audit, selected_images={c: [r["stem"] for r in rs] for c, rs in groups.items()})
    key = sc.digest(protocol)
    directory = ROOT / "cache/train_runs" / key
    sc.write_json(directory / "protocol.json", protocol)
    status(status="computing_train_scaling", run_name=name, run_key=key,
           total_groups=len(groups), completed_groups=0)
    results, compute, hits = {}, None, 0
    for index, (cls, rows) in enumerate(groups.items(), 1):
        path = directory / f"{cls}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                result = {k: saved[k] for k in saved.files}
            hits += 1
        else:
            if compute is None:
                if config["normalization"] == "global_empirical":
                    fixed = global_stats(groups, data["identity"])
                    compute = global_kernel(config, fixed)
                else:
                    compute = sc.make_kernel(config)
            result = compute(cls, rows)
            temporary = path.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream: np.savez_compressed(stream, **result)
            temporary.replace(path)
        results[cls] = result
        if index % 10 == 0 or index == len(groups):
            print(f"{name}: {index}/{len(groups)} classes, {hits} cached", flush=True)
            status(completed_groups=index, cache_hits=hits)
    summary, curves = sc.summarize(results, config)
    truncated = {c: {k: a[:, :20] for k, a in rs.items()} for c, rs in results.items()}
    summary_k20, _ = sc.summarize(truncated, config)
    outcome = dict(name=name, key=key, config=config, audit=audit, summary=summary,
                   derived_top20_summary=summary_k20,
                   mode_counts={c: int(v["f_alpha"].shape[1]) for c, v in results.items()},
                   data_errors=data["image_read_errors"])
    sc.write_json(directory / "summary.json", outcome)
    result = dict(key=key, directory=directory, config=config, audit=audit, summary=summary, curves=curves)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    figure = sc.plot_result(result)
    for axis, ticks in zip(figure.axes, ([.2, .5, 1, 2], [.2, .5, 1])):
        axis.set_xticks(ticks); axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.xaxis.set_minor_formatter(NullFormatter())
    figure.suptitle(f"TRAIN · area {config['area']} · {config['normalization']} · {len(groups)} classes")
    figure.tight_layout()
    for ext in ("png", "pdf"): figure.savefig(directory / f"scaling.{ext}")
    plt.close(figure)
    return outcome


def write_report(outcomes):
    lines = ["# ImageNet train scaling — source-based condition comparison", "",
             "All predeclared configurations are reported. A close coefficient is not proof that it was the historical run.", "",
             "Top K is capped by each group's sample count. No class is filtered based on its measured exponent.", "",
             "| Run | Classes | Images | K | Mean class-median-curve amplitude / extent | Median of all mode exponents amplitude / extent |", 
             "|---|---:|---:|---:|---|---|"]
    for r in outcomes:
        for k, key in [(100, "summary"), (20, "derived_top20_summary")]:
            a, e = r[key]["amplitude"], r[key]["extent"]
            lines.append(f"| {r['name']} | {r['audit']['computed_groups']} | {r['audit']['computed_images']} | {k} | "
                         f"{a['mean_group_median_curve_exponent']:.6f} / {e['mean_group_median_curve_exponent']:.6f} | "
                         f"{a['pooled_mode_median_exponent']:.6f} / {e['pooled_mode_median_exponent']:.6f} |")
    lines += ["", "Reference rebuttal values: amplitude 1.59, extent 1.12. Full R², settings, image IDs, mode counts and arrays are in each run's directory.", "",
              "The primary case is 45–55%, class-wise fixed RGB normalization, aligned single bbox, first 200 sorted annotation classes, at least 20 selected images/class, cap 300, K=100. The other cases reflect the existing UI's 40–60% default and global-normalization option; K=20 is derived from the same stored spectra because it is the original UI default. Sweeps remain amplitude 0.2–2.0 and extent 0.2–1.0.", ""]
    for r in outcomes:
        p = ROOT / "cache/train_runs" / r["key"]
        lines.append(f"- [{r['name']}]({(p/'summary.json').as_posix()}) · [plot]({(p/'scaling.png').as_posix()})")
    (ROOT.parent / "output/imagenet_train_condition_results.md").write_text("\n".join(lines), encoding="utf-8")
    sc.write_json(ROOT / "cache/train_study_results.json", outcomes)


def main():
    with (ROOT / "train_scaling.lock").open("a+b") as process_lock:
        process_lock.seek(0)
        if not process_lock.read(1): process_lock.write(b"0"); process_lock.flush()
        process_lock.seek(0)
        msvcrt.locking(process_lock.fileno(), msvcrt.LK_NBLCK, 1)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            cat = catalog()
            wait_for_download()
            extract_annotated_subset(cat)
            data = scan_train(cat)
            outcomes = []
            for case in STUDY:
                config = {**sc.DEFAULTS, **{k: v for k, v in case.items() if k != "name"},
                          "grouping": "classwise", "min_class_images": 20}
                outcomes.append(compute_run(config, data, case["name"]))
                write_report(outcomes)
            status(status="ready_for_review", completed_runs=len(outcomes),
                   report=str(ROOT.parent / "output/imagenet_train_condition_results.md"))
            print("Predeclared train study finished. Results ready for scientific review.", flush=True)
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        main()
    except Exception as exc:
        status(status="failed", error=repr(exc))
        raise
