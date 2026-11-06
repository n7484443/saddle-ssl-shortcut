import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # $f(i)=\sum_j\langle v_i^{o},v_j\rangle^{2}\gamma_j$ over many classes

    $$x=x_{bg}+\alpha x_o(e),\qquad
      \mathbb E[xx^\top]=\sum_j\gamma_jv_jv_j^\top,\qquad
      \mathbb E[x_ox_o^\top]=\sum_i\gamma^{o}_iv^{o}_iv^{o\top}_i$$

    Two **empirical** normalization modes are available.

    - **Global empirical mean/std (default):** pool all currently selected images across
      all classes, estimate one dataset-wide RGB mean/std, and keep those statistics fixed
      for every alpha and extent condition.
    - **Class-wise empirical mean/std (optional):** estimate one RGB mean/std separately
      for each class from that class's selected raw images. Each class keeps its own
      statistics fixed for every alpha and extent condition.

    For either mode, statistics are computed from the raw transformed pixels:

    $$\mu_c=\frac{1}{NHW}\sum_{n,p}x_{n,c,p},\qquad
      \sigma_c^2=\frac{1}{NHW}\sum_{n,p}x_{n,c,p}^2-\mu_c^2.$$

    Normalization is applied before constructing the object/background components:

    $$\tilde x=\frac{x-\mu}{\sigma},\qquad
      x=x_{bg}+\alpha x_o(e).$$

    Because $\mu,\sigma$ are fixed across the sweep, changing alpha does not change
    the object eigensystem. Extent still changes $x_o(e)$, so the object eigensystem is
    recomputed for every extent as in the original code.

    $$f(i)=\frac{\|C^\top u_i^o\|^2}{\gamma_i^o},\qquad
      C=G_{bo}^\top+\alpha G_{oo}.$$
    """)
    return


@app.cell
def _():
    import math
    import multiprocessing as mp
    import os
    import pickle
    import xml.etree.ElementTree as ET
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    torch.backends.cuda.matmul.allow_tf32 = False

    CROP = 224
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Keep raw [0, 1] pixels here. Empirical normalization is applied in BaseClassDataset.
    TRANSFORM = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(CROP),
        transforms.ToTensor(),
    ])
    return (
        CROP,
        DEVICE,
        DataLoader,
        Dataset,
        ET,
        F,
        Image,
        Path,
        TRANSFORM,
        math,
        mp,
        np,
        os,
        pickle,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(DEVICE, mo):
    ui_img = mo.ui.text(
        "/home/dataset/imagenet/ILSVRC/Data/CLS-LOC/train", label="image dir", full_width=True
    )
    ui_bbox = mo.ui.text(
        "/home/dataset/imagenet/ILSVRC/Annotations/CLS-LOC/train",
        label="bbox dir", full_width=True,
    )

    ui_nclass = mo.ui.slider(1, 200, value=200, step=1, label="classes", show_value=True)
    ui_n = mo.ui.slider(20, 2000, value=300, step=10, label="images / class", show_value=True)
    ui_res = mo.ui.slider(28, 224, value=224, step=28, label="resolution", show_value=True)
    ui_k = mo.ui.slider(2, 300, value=20, step=2, label="k modes", show_value=True)
    ui_area = mo.ui.range_slider(
        0.0, 1.0, value=[0.4, 0.6], step=0.05, label="object area fraction", show_value=True
    )
    ui_single = mo.ui.switch(value=True, label="single-bbox only")
    ui_align = mo.ui.switch(value=True, label="centre the bbox (translation only)")
    ui_class_norm = mo.ui.switch(
        value=False,
        label="class-wise mean/std (off = global empirical)",
    )

    ui_a_lo = mo.ui.number(0.0, 20.0, value=0.2, step=0.1, label="α min")
    ui_a_hi = mo.ui.number(0.1, 20.0, value=2.0, step=0.1, label="α max")
    ui_a_n = mo.ui.slider(3, 30, value=10, step=1, label="α steps", show_value=True)
    ui_e_lo = mo.ui.number(0.05, 1.0, value=0.2, step=0.05, label="e min")
    ui_e_hi = mo.ui.number(0.05, 1.0, value=1.0, step=0.05, label="e max")
    ui_e_n = mo.ui.slider(3, 20, value=9, step=1, label="e steps", show_value=True)

    ui_bs = mo.ui.slider(16, 256, value=64, step=16, label="batch", show_value=True)
    ui_workers = mo.ui.slider(0, 16, value=8, step=1, label="workers", show_value=True)

    mo.vstack([
        mo.md(f"### Config · `{DEVICE}`"),
        ui_img,
        ui_bbox,
        mo.hstack([ui_nclass, ui_n, ui_res, ui_k], justify="start"),
        mo.hstack([ui_area, ui_single, ui_align, ui_class_norm], justify="start"),
        mo.hstack([ui_a_lo, ui_a_hi, ui_a_n], justify="start"),
        mo.hstack([ui_e_lo, ui_e_hi, ui_e_n], justify="start"),
        mo.hstack([ui_bs, ui_workers], justify="start"),
    ])
    return (
        ui_a_hi,
        ui_a_lo,
        ui_a_n,
        ui_align,
        ui_area,
        ui_bbox,
        ui_bs,
        ui_class_norm,
        ui_e_hi,
        ui_e_lo,
        ui_e_n,
        ui_img,
        ui_k,
        ui_n,
        ui_nclass,
        ui_res,
        ui_single,
        ui_workers,
    )


@app.cell
def _(np, ui_a_hi, ui_a_lo, ui_a_n, ui_e_hi, ui_e_lo, ui_e_n):
    ALPHAS = np.linspace(float(ui_a_lo.value), float(ui_a_hi.value), int(ui_a_n.value))
    EXTENTS = np.linspace(float(ui_e_lo.value), float(ui_e_hi.value), int(ui_e_n.value))
    A_REF = float(ALPHAS[np.argmin(abs(ALPHAS - 1.0))])
    E_REF = float(EXTENTS[-1])
    return ALPHAS, A_REF, EXTENTS, E_REF


@app.cell(hide_code=True)
def _(mo):
    scan_btn = mo.ui.run_button(label="1 · scan")
    scan_btn
    return (scan_btn,)


@app.cell(hide_code=True)
def _(
    CROP,
    ET,
    Image,
    Path,
    mo,
    np,
    pickle,
    scan_btn,
    ui_area,
    ui_bbox,
    ui_img,
    ui_n,
    ui_nclass,
    ui_single,
):
    mo.stop(not scan_btn.value, mo.md("*idle*"))

    _broot, _iroot = Path(ui_bbox.value), Path(ui_img.value)
    _subdirs = sorted(p.name for p in _broot.iterdir() if p.is_dir())
    _classes = _subdirs[: int(ui_nclass.value)] if _subdirs else [""]

    _cache = Path(f"scan_cache_mc_{len(_classes)}.pkl")
    by_class = None
    if _cache.exists():
        try:
            _got = pickle.load(open(_cache, "rb"))
            if isinstance(_got, dict) and _got.get("key") == [str(_iroot), str(_broot), _classes]:
                by_class = _got["by_class"]
        except Exception:
            by_class = None

    if by_class is None:
        by_class = {}
        with mo.status.progress_bar(
            total=len(_classes), title="Scan · overall", subtitle="classes", show_eta=True, show_rate=True
        ) as _overall:
            for _ci, _cls in enumerate(_classes, 1):
                _bd, _idir = (_broot / _cls, _iroot / _cls) if _cls else (_broot, _iroot)
                _xml = {p.stem: p for p in _bd.rglob("*.xml")}
                _items = list(_xml.items())
                _recs = []

                with mo.status.progress_bar(
                    total=len(_items),
                    title="Scan · current class",
                    subtitle=f"{_cls or 'root'} · annotations",
                    show_eta=True,
                    show_rate=True,
                ) as _partial:
                    for _stem, _xp in _items:
                        _ip = None
                        for _ext in (".JPEG", ".jpg", ".JPG", ".png", ".PNG"):
                            _c = _idir / f"{_stem}{_ext}"
                            if _c.exists():
                                _ip = _c
                                break

                        if _ip is not None:
                            try:
                                with Image.open(_ip) as _im:
                                    _w, _h = _im.size
                                _s = 256 / min(_w, _h)
                                _l, _t = (round(_w * _s) - CROP) // 2, (round(_h * _s) - CROP) // 2
                                _boxes = []

                                for _obj in ET.parse(_xp).getroot().findall("object"):
                                    _b = _obj.find("bndbox")
                                    _x0, _y0, _x1, _y1 = (
                                        int(_b.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")
                                    )
                                    _x0 = max(0.0, min(CROP, _x0 * _s - _l))
                                    _x1 = max(0.0, min(CROP, _x1 * _s - _l))
                                    _y0 = max(0.0, min(CROP, _y0 * _s - _t))
                                    _y1 = max(0.0, min(CROP, _y1 * _s - _t))
                                    if _x1 - _x0 >= 4 and _y1 - _y0 >= 4:
                                        _boxes.append([round(_x0), round(_y0), round(_x1), round(_y1)])

                                if _boxes:
                                    _m = np.zeros((CROP, CROP), np.float32)
                                    for _bx in _boxes:
                                        _m[_bx[1] : _bx[3], _bx[0] : _bx[2]] = 1.0
                                    _recs.append(
                                        {"path": str(_ip), "boxes": _boxes, "area": float(_m.mean())}
                                    )
                            except Exception:
                                pass

                        _partial.update(increment=1, subtitle=f"{_cls or 'root'} · {_stem}")

                by_class[_cls] = _recs
                _overall.update(increment=1, subtitle=f"class {_ci}/{len(_classes)} · {_cls or 'root'}")

        pickle.dump(
            {"key": [str(_iroot), str(_broot), _classes], "by_class": by_class},
            open(_cache, "wb"),
        )

    _lo, _hi = ui_area.value
    CLASS_RECORDS = {}
    for _cls, _recs in by_class.items():
        _sel = [r for r in _recs if _lo <= r["area"] <= _hi]
        if ui_single.value:
            _sel = [r for r in _sel if len(r["boxes"]) == 1]
        if len(_sel) > ui_n.value:
            _idx = sorted(np.random.default_rng(0).permutation(len(_sel))[: ui_n.value])
            _sel = [_sel[i] for i in _idx]
        if len(_sel) >= 20:
            CLASS_RECORDS[_cls] = _sel

    _sizes = [len(v) for v in CLASS_RECORDS.values()]
    mo.md(
        f"**{len(CLASS_RECORDS)}** classes kept · images/class "
        f"min **{min(_sizes)}**, median **{int(np.median(_sizes))}**, max **{max(_sizes)}** · "
        f"total **{sum(_sizes)}**"
    )
    return (CLASS_RECORDS,)


@app.cell(hide_code=True)
def _(CROP, Dataset, F, Image, TRANSFORM, torch, ui_align):
    class BaseClassDataset(Dataset):
        """Load/normalize/align each image once. Extent-specific objects are built later."""

        def __init__(self, records, norm_mean, norm_std):
            self.records = records
            self.norm_mean = norm_mean
            self.norm_std = norm_std

        def __len__(self):
            return len(self.records)

        def __getitem__(self, n):
            rec = self.records[n]
            x = TRANSFORM(Image.open(rec["path"]).convert("RGB"))
            x = (x - self.norm_mean) / self.norm_std

            x0, y0, x1, y1 = rec["boxes"][0]
            if ui_align.value:
                theta = torch.tensor([
                    [1.0, 0.0, (x0 + x1) / CROP - 1.0],
                    [0.0, 1.0, (y0 + y1) / CROP - 1.0],
                ]).unsqueeze(0)
                grid = F.affine_grid(theta, (1, 3, CROP, CROP), align_corners=False)
                x = F.grid_sample(x.unsqueeze(0), grid, align_corners=False)[0]
                w, h = x1 - x0, y1 - y0
                bx, by = (CROP - w) // 2, (CROP - h) // 2
            else:
                bx, by, w, h = x0, y0, x1 - x0, y1 - y0

            geom = torch.tensor([bx, by, w, h], dtype=torch.int32)
            return x, geom

    return (BaseClassDataset,)


@app.cell(hide_code=True)
def _(mo):
    run_btn = mo.ui.run_button(label="2 · collect and compute f")
    run_btn
    return (run_btn,)


@app.cell(hide_code=True)
def _(
    ALPHAS,
    A_REF,
    BaseClassDataset,
    CLASS_RECORDS,
    CROP,
    DEVICE,
    DataLoader,
    Dataset,
    EXTENTS,
    E_REF,
    F,
    Image,
    TRANSFORM,
    math,
    mo,
    mp,
    np,
    os,
    run_btn,
    torch,
    ui_bs,
    ui_class_norm,
    ui_k,
    ui_res,
    ui_workers,
):
    mo.stop(not run_btn.value, mo.md("*idle*"))

    _use_class_norm = bool(ui_class_norm.value)
    _res = int(ui_res.value)
    _nw = min(ui_workers.value, max(0, (os.cpu_count() or 1) - 2))
    _ctx = {"multiprocessing_context": mp.get_context("fork")} if _nw > 0 else {}
    _nclass = len(CLASS_RECORDS)

    class _RawStatsDataset(Dataset):
        def __init__(self, records):
            self.records = records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, n):
            return TRANSFORM(Image.open(self.records[n]["path"]).convert("RGB"))

    def _global_empirical_stats(_overall):
        """One pooled RGB mean/std from all currently selected raw images."""
        _sum = torch.zeros(3, dtype=torch.float64)
        _sumsq = torch.zeros(3, dtype=torch.float64)
        _count = 0
        _total_images = sum(len(v) for v in CLASS_RECORDS.values())
        _done = 0

        with mo.status.progress_bar(
            total=_total_images,
            title="Current · empirical normalization stats",
            subtitle="selected images",
            show_eta=True,
            show_rate=True,
        ) as _partial:
            for _ci, (_cls, _recs) in enumerate(CLASS_RECORDS.items(), 1):
                _loader = DataLoader(
                    _RawStatsDataset(_recs),
                    batch_size=ui_bs.value,
                    num_workers=_nw,
                    pin_memory=False,
                    **_ctx,
                )
                for _batch in _loader:
                    # This is only one pass over the selected subset; keep the original
                    # FP64 accumulation for stable empirical statistics.
                    _batch = _batch.double()
                    _bsz = _batch.shape[0]
                    _sum += _batch.sum(dim=(0, 2, 3))
                    _sumsq += (_batch * _batch).sum(dim=(0, 2, 3))
                    _count += _bsz * CROP * CROP
                    _done += _bsz
                    _partial.update(
                        increment=_bsz,
                        subtitle=f"class {_ci}/{_nclass} · {_cls or 'root'} · {_done}/{_total_images} images",
                    )

                _overall.update(increment=1, subtitle=f"empirical stats · class {_ci}/{_nclass}")

        _mean = _sum / float(_count)
        _var = _sumsq / float(_count) - _mean * _mean
        _std = torch.sqrt(_var.clamp_min(1e-12))
        return _mean.float().view(3, 1, 1), _std.float().view(3, 1, 1)

    def _class_empirical_stats(_recs, _ci, _cls):
        """One RGB mean/std from the selected raw images of a single class."""
        _sum = torch.zeros(3, dtype=torch.float64)
        _sumsq = torch.zeros(3, dtype=torch.float64)
        _count = 0
        _done = 0

        _loader = DataLoader(
            _RawStatsDataset(_recs),
            batch_size=ui_bs.value,
            num_workers=_nw,
            pin_memory=False,
            **_ctx,
        )

        with mo.status.progress_bar(
            total=len(_recs),
            title="Current · class-wise normalization stats",
            subtitle=f"class {_ci}/{_nclass} · {_cls or 'root'}",
            show_eta=True,
            show_rate=True,
        ) as _partial:
            for _batch in _loader:
                _batch = _batch.double()
                _bsz = _batch.shape[0]
                _sum += _batch.sum(dim=(0, 2, 3))
                _sumsq += (_batch * _batch).sum(dim=(0, 2, 3))
                _count += _bsz * CROP * CROP
                _done += _bsz
                _partial.update(
                    increment=_bsz,
                    subtitle=f"class {_ci}/{_nclass} · {_done}/{len(_recs)} images",
                )

        _mean = _sum / float(_count)
        _var = _sumsq / float(_count) - _mean * _mean
        _std = torch.sqrt(_var.clamp_min(1e-12))
        return _mean.float().view(3, 1, 1), _std.float().view(3, 1, 1)

    def _load_class_once(_recs, _norm_mean, _norm_std):
        """JPEG decode + resize/crop + normalization + alignment exactly once per class."""
        _loader = DataLoader(
            BaseClassDataset(_recs, _norm_mean, _norm_std),
            batch_size=ui_bs.value,
            num_workers=_nw,
            pin_memory=DEVICE.type == "cuda",
            **_ctx,
        )
        _xs, _geoms = [], []
        for _x, _geom in _loader:
            _xs.append(_x.float())
            _geoms.append(_geom)

        _X = torch.cat(_xs, dim=0).to(DEVICE, non_blocking=True)
        _geom = torch.cat(_geoms, dim=0).cpu()
        return _X, _geom

    def _full_mask(_geom):
        """Vectorized full-bbox mask for a cached class."""
        _bx = _geom[:, 0].view(-1, 1, 1)
        _by = _geom[:, 1].view(-1, 1, 1)
        _w = _geom[:, 2].view(-1, 1, 1)
        _h = _geom[:, 3].view(-1, 1, 1)
        _xx = torch.arange(CROP).view(1, 1, CROP)
        _yy = torch.arange(CROP).view(1, CROP, 1)
        _mask = (_xx >= _bx) & (_xx < _bx + _w) & (_yy >= _by) & (_yy < _by + _h)
        return _mask.unsqueeze(1).to(device=DEVICE, non_blocking=True)

    def _downsample_flat(_x):
        if _res != CROP:
            _x = F.interpolate(_x, size=(_res, _res), mode="area")
        return _x.reshape(_x.shape[0], -1)

    def _object_at_extent(_X, _geom, _full, _extent):
        """Build only the extent-dependent object from the cached aligned images."""
        if float(_extent) >= 1.0:
            return _downsample_flat(_X * _full)

        _obj = torch.zeros_like(_X)
        _lin = math.sqrt(float(_extent))

        # Geometry varies by image, so only this small resize/place operation remains
        # per image. The expensive JPEG/transform/alignment path is no longer repeated.
        for _i in range(_X.shape[0]):
            _bx, _by, _w, _h = [int(v) for v in _geom[_i].tolist()]
            _sw = max(2, int(round(_w * _lin)))
            _sh = max(2, int(round(_h * _lin)))
            _px = _bx + (_w - _sw) // 2
            _py = _by + (_h - _sh) // 2
            _patch = F.interpolate(
                _X[_i : _i + 1, :, _by : _by + _h, _bx : _bx + _w],
                size=(_sh, _sw),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            _obj[_i, :, _py : _py + _sh, _px : _px + _sw] = _patch[0]

        return _downsample_flat(_obj)

    def _condition_eigensystem(_Xo, _Xb, _alpha):
        """Keep Gram matrices/eigendecomposition on DEVICE; copy out only final summaries."""
        _n = _Xo.shape[0]
        _Goo = (_Xo @ _Xo.T).double() / _n
        _Goo = 0.5 * (_Goo + _Goo.T)
        _GboT = (_Xo @ _Xb.T).double() / _n
        _C = _GboT + float(_alpha) * _Goo

        _lo, _Uo = torch.linalg.eigh(_Goo)
        _lo, _Uo = _lo.flip(0).clamp_min(1e-300), _Uo.flip(1)
        _kk = min(int(ui_k.value), _lo.numel())
        _lo, _Uo = _lo[:_kk], _Uo[:, :_kk]

        _f = (((_C.T @ _Uo) ** 2).sum(0) / _lo)
        _leak = float((_GboT.norm() / _Goo.norm()).item())
        return _f.detach().cpu().numpy(), _lo, _Uo, _Goo, _GboT, _leak

    # Both normalization modes require one raw-image statistics pass. Global mode pools
    # all classes; class-wise mode computes a separate fixed pair for each class.
    _stats_units = _nclass
    _compute_units = _nclass * (1 + len(EXTENTS) + len(ALPHAS))
    _total_units = _stats_units + _compute_units

    F_ALPHA, F_EXT, GAMMA, LEAK = {}, {}, {}, {}
    _class_norm_stats = {}

    with mo.status.progress_bar(
        total=_total_units,
        title="Compute · overall",
        subtitle="starting",
        show_eta=True,
        show_rate=True,
    ) as _overall:
        if not _use_class_norm:
            _global_mean, _global_std = _global_empirical_stats(_overall)

        for _ci, (_cls, _recs) in enumerate(CLASS_RECORDS.items(), 1):
            if _use_class_norm:
                _norm_mean, _norm_std = _class_empirical_stats(_recs, _ci, _cls)
                _class_norm_stats[_cls] = (
                    _norm_mean.detach().cpu(),
                    _norm_std.detach().cpu(),
                )
                _overall.update(
                    increment=1,
                    subtitle=f"class-wise stats · class {_ci}/{_nclass}",
                )
            else:
                _norm_mean, _norm_std = _global_mean, _global_std

            _f_ext = []
            _gam_ref = _uo_ref = _Goo_ref = _GboT_ref = None
            _eref_leak = None
            _class_total = 1 + len(EXTENTS) + len(ALPHAS)

            with mo.status.progress_bar(
                total=_class_total,
                title="Current · class computation",
                subtitle=f"class {_ci}/{_nclass} · loading/cache",
                show_eta=True,
                show_rate=True,
            ) as _partial:
                # Main speedup: decode / transform / normalize / align each image once.
                _X, _geom = _load_class_once(_recs, _norm_mean, _norm_std)
                _full = _full_mask(_geom)
                _Xb = _downsample_flat(_X * (~_full))

                _partial.update(increment=1, subtitle=f"class {_ci}/{_nclass} · cached {len(_recs)} images")
                _overall.update(increment=1, subtitle=f"class {_ci}/{_nclass} · image cache ready")

                # Extent changes only the object. Background and preprocessed images are reused.
                for _ei, _e in enumerate(EXTENTS, 1):
                    _ekey = float(_e)
                    _Xo = _object_at_extent(_X, _geom, _full, _ekey)
                    _f, _gam, _Uo, _Goo, _GboT, _leak = _condition_eigensystem(_Xo, _Xb, A_REF)
                    _f_ext.append(_f)

                    if np.isclose(_ekey, E_REF):
                        _gam_ref, _uo_ref, _Goo_ref, _GboT_ref = _gam, _Uo, _Goo, _GboT
                        _eref_leak = _leak

                    del _Xo
                    _partial.update(
                        increment=1,
                        subtitle=f"class {_ci}/{_nclass} · extent {_ei}/{len(EXTENTS)} · e={_ekey:.3f}",
                    )
                    _overall.update(
                        increment=1,
                        subtitle=f"class {_ci}/{_nclass} · extent {_ei}/{len(EXTENTS)}",
                    )

                _k = min(len(a) for a in _f_ext)
                F_EXT[_cls] = np.stack([a[:_k] for a in _f_ext])

                # The alpha sweep only needs the reference Gram/eigenbasis, so release
                # the image-sized cache before the cheap alpha loop.
                del _X, _geom, _full, _Xb

                # Fixed normalization: alpha changes C only; no image work and no new eigensystem.
                _fa = []
                for _ai, _a in enumerate(ALPHAS, 1):
                    _akey = float(_a)
                    _C = _GboT_ref + _akey * _Goo_ref
                    _f_a = (((_C.T @ _uo_ref) ** 2).sum(0) / _gam_ref).detach().cpu().numpy()
                    _fa.append(_f_a[:_k])

                    _partial.update(
                        increment=1,
                        subtitle=f"class {_ci}/{_nclass} · alpha {_ai}/{len(ALPHAS)} · α={_akey:.3f}",
                    )
                    _overall.update(
                        increment=1,
                        subtitle=f"class {_ci}/{_nclass} · alpha {_ai}/{len(ALPHAS)}",
                    )

            F_ALPHA[_cls] = np.stack(_fa)
            GAMMA[_cls] = _gam_ref[:_k].detach().cpu().numpy()
            LEAK[_cls] = _eref_leak

            del _gam_ref, _uo_ref, _Goo_ref, _GboT_ref
            # Do not empty the CUDA allocator inside the extent loop. Once per class is enough
            # to cap memory while still allowing reuse during the expensive inner loop.
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    if _use_class_norm:
        _norm_name = "class-wise empirical mean/std"
        _means = torch.stack([m.view(3) for m, _ in _class_norm_stats.values()])
        _stds = torch.stack([s.view(3) for _, s in _class_norm_stats.values()])
        _mean_vals = [float(v) for v in _means.median(dim=0).values]
        _std_vals = [float(v) for v in _stds.median(dim=0).values]
        _stats_label = "median class mean/std"
    else:
        _norm_name = "global empirical mean/std (default)"
        _mean_vals = [float(v) for v in _global_mean.view(-1)]
        _std_vals = [float(v) for v in _global_std.view(-1)]
        _stats_label = "global mean/std"

    mo.md(
        f"done · **{len(F_ALPHA)}** classes · k=**{min(len(g) for g in GAMMA.values())}** · "
        f"normalization = **{_norm_name}** · "
        f"{_stats_label}: mean=**[{_mean_vals[0]:.4f}, {_mean_vals[1]:.4f}, {_mean_vals[2]:.4f}]** · "
        f"std=**[{_std_vals[0]:.4f}, {_std_vals[1]:.4f}, {_std_vals[2]:.4f}]** · "
        f"median leakage ‖G_bo‖/‖G_oo‖ = **{np.median(list(LEAK.values())):.3f}**"
    )
    return F_ALPHA, F_EXT


@app.cell(hide_code=True)
def _(ALPHAS, A_REF, EXTENTS, E_REF, F_ALPHA, F_EXT, mo, np):
    mo.stop(not F_ALPHA, mo.md(""))

    _ia = int(np.argmin(abs(ALPHAS - A_REF)))
    _ie = int(np.argmin(abs(EXTENTS - E_REF)))

    RATIO_A, RATIO_E = [], []
    for _c in F_ALPHA:
        _fa, _fe = F_ALPHA[_c], F_EXT[_c]
        RATIO_A.append(_fa / _fa[_ia])
        RATIO_E.append(_fe / _fe[_ie])

    RATIO_A = np.concatenate(RATIO_A, axis=1)
    RATIO_E = np.concatenate(RATIO_E, axis=1)

    def _fit_each_power(_x, _ratio, _xref):
        """Fit log(f/f_ref) = a + k log(x/x_ref) for every (class, mode/rank) curve."""
        _lx = np.log(np.asarray(_x, dtype=float) / float(_xref))
        _k, _a, _r2 = [], [], []

        for _j in range(_ratio.shape[1]):
            _y = _ratio[:, _j]
            _ok = np.isfinite(_lx) & np.isfinite(_y) & (_y > 0)
            if _ok.sum() < 3 or np.var(_lx[_ok]) == 0:
                _k.append(np.nan)
                _a.append(np.nan)
                _r2.append(np.nan)
                continue

            _xx, _yy = _lx[_ok], np.log(_y[_ok])
            _kj, _aj = np.polyfit(_xx, _yy, 1)
            _pred = _aj + _kj * _xx
            _ss_res = np.sum((_yy - _pred) ** 2)
            _ss_tot = np.sum((_yy - _yy.mean()) ** 2)
            _r2j = 1.0 - _ss_res / _ss_tot if _ss_tot > 0 else np.nan

            _k.append(_kj)
            _a.append(_aj)
            _r2.append(_r2j)

        return np.asarray(_k), np.asarray(_a), np.asarray(_r2)

    K_A, INT_A, R2_A = _fit_each_power(ALPHAS, RATIO_A, A_REF)
    K_E, INT_E, R2_E = _fit_each_power(EXTENTS, RATIO_E, E_REF)

    def _summary(_k, _a, _r2, _target):
        _ok = np.isfinite(_k) & np.isfinite(_a) & np.isfinite(_r2)
        _kk, _aa, _rr = _k[_ok], _a[_ok], _r2[_ok]
        return {
            "n": len(_kk),
            "k_med": np.median(_kk),
            "k_q": np.quantile(_kk, [0.25, 0.75]),
            "mae_k": np.mean(np.abs(_kk - _target)),
            "med_abs_kerr": np.median(np.abs(_kk - _target)),
            "r2_med": np.median(_rr),
            "r2_q": np.quantile(_rr, [0.25, 0.75]),
            "abs_a_med": np.median(np.abs(_aa)),
        }

    STAT_A = _summary(K_A, INT_A, R2_A, 2.0)
    STAT_E = _summary(K_E, INT_E, R2_E, 1.0)

    mo.md(
        f"""
        Power-law fits: $\\log(f/f_{{ref}})=a+k\\log(x/x_{{ref}})$

        - amplitude target: $k=2$ · median fitted $k={STAT_A['k_med']:.4f}$
        - extent target: $k=1$ · median fitted $k={STAT_E['k_med']:.4f}$
        """
    )
    return K_A, K_E, R2_A, R2_E, RATIO_A, RATIO_E, STAT_A, STAT_E


@app.cell(hide_code=True)
def _(
    ALPHAS,
    A_REF,
    EXTENTS,
    E_REF,
    K_A,
    K_E,
    R2_A,
    R2_E,
    RATIO_A,
    RATIO_E,
    STAT_A,
    STAT_E,
    mo,
    np,
    plt,
):
    _fig, _ax = plt.subplots(2, 2, figsize=(11.5, 8.4))

    for _ax_i, _x, _r, _lab, _ref, _title in [
        (
            _ax[0, 0], ALPHAS, RATIO_A, r"$\alpha$", (ALPHAS / A_REF) ** 2,
            r"amplitude: target $(\alpha/\alpha_{ref})^2$",
        ),
        (
            _ax[0, 1], EXTENTS, RATIO_E, "extent", EXTENTS / E_REF,
            r"extent: target $e/e_{ref}$",
        ),
    ]:
        _ax_i.plot(_x, _r, lw=0.4, alpha=0.15, color="tab:blue")
        _ax_i.fill_between(
            _x,
            np.nanquantile(_r, 0.25, axis=1),
            np.nanquantile(_r, 0.75, axis=1),
            alpha=0.3,
            color="tab:blue",
        )
        _ax_i.plot(_x, np.nanmedian(_r, axis=1), "o-", color="k", lw=2, label="median")
        _ax_i.plot(_x, _ref, "--", color="crimson", lw=1.2, label="target")
        _ax_i.set(xlabel=_lab, ylabel=r"$f/f_{ref}$", title=_title)
        _ax_i.grid(alpha=0.3)
        _ax_i.legend(fontsize=8)

    for _ax_i, _k, _r2, _target, _kmed, _title in [
        (_ax[1, 0], K_A, R2_A, 2.0, STAT_A["k_med"], "amplitude power-law fits"),
        (_ax[1, 1], K_E, R2_E, 1.0, STAT_E["k_med"], "extent power-law fits"),
    ]:
        _ok = np.isfinite(_k) & np.isfinite(_r2)
        _ax_i.scatter(_k[_ok], _r2[_ok], s=8, alpha=0.4)
        _ax_i.axvline(_target, ls="--", color="crimson", lw=1.2, label=f"target k={_target:g}")
        _ax_i.axvline(_kmed, ls="-", color="k", lw=1.5, label=f"median k={_kmed:.3f}")
        _ax_i.axhline(1.0, ls=":", color="k", lw=1.0)
        _ax_i.set(xlabel="fitted exponent k", ylabel=r"$R^2$ in log-log fit", title=_title)
        _ax_i.grid(alpha=0.3)
        _ax_i.legend(fontsize=8)

    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md(
            f"""
            | statistic | amplitude | extent |
            |---|---:|---:|
            | target exponent | 2.000 | 1.000 |
            | median exponent $k$ | {STAT_A["k_med"]:.4f} | {STAT_E["k_med"]:.4f} |
            | $k$ IQR | [{STAT_A["k_q"][0]:.4f}, {STAT_A["k_q"][1]:.4f}] | [{STAT_E["k_q"][0]:.4f}, {STAT_E["k_q"][1]:.4f}] |
            | median $|k-k_{{target}}|$ | {STAT_A["med_abs_kerr"]:.4f} | {STAT_E["med_abs_kerr"]:.4f} |
            | mean $|k-k_{{target}}|$ | {STAT_A["mae_k"]:.4f} | {STAT_E["mae_k"]:.4f} |
            | median $R^2$ | {STAT_A["r2_med"]:.4f} | {STAT_E["r2_med"]:.4f} |
            | $R^2$ IQR | [{STAT_A["r2_q"][0]:.4f}, {STAT_A["r2_q"][1]:.4f}] | [{STAT_E["r2_q"][0]:.4f}, {STAT_E["r2_q"][1]:.4f}] |
            | median $|a|$ | {STAT_A["abs_a_med"]:.4f} | {STAT_E["abs_a_med"]:.4f} |
            | valid (class, mode/rank) fits | {STAT_A["n"]} | {STAT_E["n"]} |
            """
        ),
    ])
    return


if __name__ == "__main__":
    app.run()
