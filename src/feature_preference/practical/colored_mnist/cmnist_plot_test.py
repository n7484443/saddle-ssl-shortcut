import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    return np, plt, sns


@app.cell
def _(plt, sns):
    sns.set_theme(style="whitegrid", font="cmr10", font_scale=1.9)
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    colors = sns.color_palette("dark")
    colors
    return (colors,)


@app.cell
def _(np):
    import os

    def load_from_csv(path):
        output = []
        if not os.path.exists(path):
            return np.array(output)
        with open(path, 'r') as f:
            for line in f:
                output.append(float(line.strip()))
        return np.array(output)

    def load_from_csv_split(path):
        output = []
        if not os.path.exists(path):
            return np.array(output)
        with open(path, 'r') as f:
            for line in f:
                split = line.strip().split(',')
                if len(output) == 0:
                    for i in range(len(split)):
                        output.append([])
                for i in range(len(split)):
                    output[i].append(float(split[i].strip()))
        for i in range(len(output)):
            output[i] = np.array(output[i])
        return output

    return load_from_csv, load_from_csv_split, os


@app.cell
def _(load_from_csv, load_from_csv_split):
    ratios = [0.05, 0.1, 0.15]
    lambdas = 0.05
    loss = []
    acc = []
    acc_types = []
    step_list = []
    eig_list = []
    seed = 0
    directory = f"datas/lambda_{lambdas}_{seed}"
    for _r in ratios:
        path = f"{_r}"
        loss_array = load_from_csv(f"{directory}/{path}/loss.csv")
        acc_array = load_from_csv(f"{directory}/{path}/accuracy.csv")
        steps_lists_array = load_from_csv(f"{directory}/{path}/steps.csv")
        eigvals_lists_array = load_from_csv_split(f"{directory}/{path}/eig.csv")
        loss.append(loss_array)
        acc.append(acc_array)
        step_list.append(steps_lists_array)
        eig_list.append(eigvals_lists_array)
    return acc, ratios, step_list


@app.cell
def _(acc, colors, np, plt, ratios, step_list):
    # NOTE: The paper caption says 70.5%, but multiplying by 1.005 below gives 70.35%. Use a consistent threshold.
    background_predictability = 70
    ESCAPE_MULT = 1.005

    fig, axs = plt.subplots(1, 1, figsize=(13, 5.3))
    epochs = len(step_list[0])

    positions = []
    ran = 350
    for r in range(len(ratios)):
        pos = np.where(acc[r] >= background_predictability * ESCAPE_MULT)
        if len(pos[0]) > 0:
            positions.append(pos[0][0])
            axs.axvline(step_list[r][positions[-1]], linestyle="--",
                        color=colors[r], linewidth=4, alpha=0.7)
        axs.plot(step_list[r][:ran], acc[r][:ran], alpha=0.6, linewidth=5,
                 color=colors[r], label=f"Object ratio : {ratios[r]:.3f}")

    axs.text(600, 68.5, "background-label correlation")
    axs.set_ylim(65, 77.5)
    axs.axhline(background_predictability, linestyle="--", color="black",
                linewidth=2, alpha=0.6)
    axs.set_xlabel("Step")
    axs.set_ylabel("Linear Probing Accuracy")
    axs.legend()

    plt.tight_layout()
    plt.savefig("figure/cmnist.pdf", format="pdf")
    plt.gca()
    return background_predictability, ran


@app.cell
def _(acc, background_predictability, np, ran, ratios, step_list):
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

    def curve_area(x, y, baseline=None, clip=False):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if baseline is not None:
            y = y - baseline
            if clip:
                y = np.clip(y, 0, None)
        return _trapz(y, x)

    CEILING = 100.0

    areas = {}
    for _i, _ratio in enumerate(ratios):
        _x, _y = step_list[_i][1:ran], acc[_i][1:ran]
        _raw = curve_area(_x, _y)
        _above = curve_area(_x, _y, baseline=background_predictability)
        _pos = curve_area(_x, _y, baseline=background_predictability, clip=True)

        _span = _x[-1] - _x[0]
        _max_above = (CEILING - background_predictability) * _span
        _rel = _above / _max_above * 100.0
        _rel_pos = _pos / _max_above
        areas[_ratio] = {
            "area": _raw, "above": _above, "above_pos": _pos,
            "rel_above": _rel, "rel_above_pos": _rel_pos,
            "mean_acc": _raw / _span, "x_range": (_x[0], _x[-1]),
        }
        print(f"ratio {_ratio:.3f} | area={_raw:12,.1f} | above70={_above:10,.1f} "
              f"| rel={_rel:6.3f} | above70(+only)={_pos:10,.1f} | rel(+only)={_rel_pos:6.3f} "
              f"| mean acc={_raw/_span:6.3f} | steps {_x[0]:.0f}~{_x[-1]:.0f}")
    return


@app.cell
def _(np):
    def first_reach(steps, y, thr, persist=1):
        """First step with acc >= thr for persist consecutive observations; nan if none exists."""
        s = np.asarray(steps, float)
        v = np.asarray(y, float)
        n = min(s.size, v.size)
        if n == 0:
            return np.nan
        s, v = s[:n], v[:n]
        ok = v >= thr
        if persist <= 1:
            idx = np.flatnonzero(ok)
            return float(s[idx[0]]) if idx.size else np.nan
        run = np.convolve(ok.astype(int), np.ones(persist, int), mode="valid")
        idx = np.flatnonzero(run == persist)
        return float(s[idx[0]]) if idx.size else np.nan

    def is_desc(vals):
        """Check whether escape steps strictly decrease as object ratio increases."""
        v = np.asarray(vals, float)
        if np.isnan(v).any():
            return False
        return bool(np.all(np.diff(v) < 0))

    return first_reach, is_desc


@app.cell
def _(acc, first_reach, is_desc, np, ratios, step_list):
    THR_LIST = [70.35, 70.5, 71.0, 72.0, 73.0]      # Absolute threshold
    BAND_LIST = [1.0, 3.0, 5.0]                     # Upward exit from the 70 +/- w band
    PERSIST_LIST = [1, 3, 5]                        # Number of consecutive observations required
    BASE = 70.0

    escape_rows = []
    for _p in PERSIST_LIST:
        for _thr in THR_LIST:
            _v = [first_reach(step_list[_i], acc[_i], _thr, _p)
                  for _i in range(len(ratios))]
            escape_rows.append(dict(kind="thr", param=_thr, persist=_p,
                                    label=f"acc >= {_thr:.2f}", vals=_v,
                                    ok=is_desc(_v)))
        for _w in BAND_LIST:
            _v = [first_reach(step_list[_i], acc[_i], BASE + _w, _p)
                  for _i in range(len(ratios))]
            escape_rows.append(dict(kind="band", param=_w, persist=_p,
                                    label=f"exit 70+-{_w:.0f}", vals=_v,
                                    ok=is_desc(_v)))

    _n_ok = sum(r["ok"] for r in escape_rows)
    print(f"{'definition':>18} {'persist':>7} "
          + " ".join(f"{r:>9.3f}" for r in ratios)
          + f" {'ratio':>7} {'order':>6}")
    for _row in escape_rows:
        _v = _row["vals"]
        _rr = (_v[0] / _v[-1]) if (not np.isnan(_v[0]) and _v[-1]) else np.nan
        print(f"{_row['label']:>18} {_row['persist']:7d} "
              + " ".join(f"{x:9.0f}" if not np.isnan(x) else f"{'nan':>9}" for x in _v)
              + f" {_rr:7.2f} {'OK' if _row['ok'] else 'BROKEN':>6}")
    print(f"\n순서 보존 셀: {_n_ok} / {len(escape_rows)}")
    print("ratio 열 = t(가장 작은 object) / t(가장 큰 object).")
    print("임계값을 바꿔도 이 비율이 안정적이면, 임계값은 escape time 을 평행이동시킬 뿐")
    print("설정 간 상대 관계를 만들어내지 않는다는 뜻.")
    return THR_LIST, escape_rows


@app.cell
def _(escape_rows, np):
    # One-line summary for the rebuttal
    _ok_rows = [r for r in escape_rows if r["ok"]]
    if _ok_rows:
        _rr = np.array([r["vals"][0] / r["vals"][-1] for r in _ok_rows], float)
        print(f"순서 보존 셀 {len(_ok_rows)}개에서")
        print(f"  t(0.05)/t(0.15) = {_rr.mean():.2f} +- {_rr.std():.2f}  "
              f"[{_rr.min():.2f}, {_rr.max():.2f}]")
        print(f"  변동계수 CV = {_rr.std() / _rr.mean() * 100:.1f}%")
    _bad = [r for r in escape_rows if not r["ok"]]
    if _bad:
        print("\n순서가 깨진 정의:")
        for _r in _bad:
            print(f"  {_r['label']}  persist={_r['persist']}  vals={_r['vals']}")
    else:
        print("\n모든 정의에서 순서 보존.")
    return


@app.cell
def _(THR_LIST, colors, escape_rows, plt, ratios):
    # Plot escape step against threshold to check whether the curves cross.
    _fig, _ax = plt.subplots(1, 1, figsize=(11, 5.3))
    for _i in range(len(ratios)):
        _y = [r["vals"][_i] for r in escape_rows
              if r["kind"] == "thr" and r["persist"] == 1]
        _ax.plot(THR_LIST, _y, "o-", lw=4, ms=11, color=colors[_i], alpha=0.8,
                 label=f"Object ratio : {ratios[_i]:.3f}")
    _ax.set_xlabel("Escape threshold (%)")
    _ax.set_ylabel("Escape step")
    _ax.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("figure/cmnist_threshold_sensitivity.pdf", format="pdf")
    print("곡선이 서로 교차하지 않으면 임계값 선택과 무관하게 순서가 보존됨.")
    plt.gca()
    return


@app.cell
def _(load_from_csv, ratios):
    SEEDS = [0, 1, 2]
    LAMBDA_F = 0.05

    acc_seeds = {}
    for _r in ratios:
        _runs = []
        for _s in SEEDS:
            _dir = f"datas/lambda_{LAMBDA_F}_{_s}/{_r}"
            _a = load_from_csv(f"{_dir}/accuracy.csv")
            _t = load_from_csv(f"{_dir}/steps.csv")
            if _a.size and _t.size:
                _n = min(_a.size, _t.size)
                _runs.append((_t[:_n], _a[:_n]))
        acc_seeds[_r] = _runs
        print(f"ratio {_r:.3f}: {len(_runs)} seed(s)")

    N_SEED_MIN = min(len(v) for v in acc_seeds.values())
    if N_SEED_MIN < 2:
        print("\n[!] 시드 1개. strict dom == mean dom 이 되고 permutation 은 skip 됩니다.")
    return (acc_seeds,)


@app.cell
def _(acc_seeds, np, ratios):
    N_GRID = 400
    STEP_LO, STEP_HI = None, None    # Set a numeric range, or use None for automatic limits.

    _lo = max(min(t[0] for t, _ in acc_seeds[r]) for r in ratios if acc_seeds[r])
    _hi = min(max(t[-1] for t, _ in acc_seeds[r]) for r in ratios if acc_seeds[r])
    GRID_LO = float(_lo if STEP_LO is None else STEP_LO)
    GRID_HI = float(_hi if STEP_HI is None else STEP_HI)
    common_grid = np.linspace(GRID_LO, GRID_HI, N_GRID)

    def on_grid(runs):
        return np.vstack([np.interp(common_grid, t, a) for t, a in runs])

    print(f"grid: {GRID_LO:.0f} ~ {GRID_HI:.0f}, {N_GRID} pts")
    return (on_grid,)


@app.cell
def _(acc_seeds, on_grid, ratios):
    print(f"{'pair':>18} {'mean dom':>9} {'strict dom':>11} {'mean gap':>9}")
    dom_rows = []
    for _a in range(len(ratios)):
        for _b in range(_a + 1, len(ratios)):
            _hi_r, _lo_r = ratios[_b], ratios[_a]
            if not acc_seeds[_hi_r] or not acc_seeds[_lo_r]:
                continue
            _H, _L = on_grid(acc_seeds[_hi_r]), on_grid(acc_seeds[_lo_r])
            _md = float((_H.mean(0) > _L.mean(0)).mean())
            _sd = float((_H.min(0) > _L.max(0)).mean())
            _gap = float((_H.mean(0) - _L.mean(0)).mean())
            dom_rows.append(dict(hi=_hi_r, lo=_lo_r,
                                 mean_dom=_md, strict_dom=_sd, gap=_gap))
            print(f"{_hi_r:.3f} vs {_lo_r:.3f}   {_md:9.3f} {_sd:11.3f} {_gap:9.3f}")
    return


@app.cell
def _(acc_seeds, np, on_grid, ratios):
    from itertools import combinations

    def perm_test(A, B):
        pooled = np.vstack([A, B])
        nA, n = len(A), len(pooled)

        def _stat(idx):
            rest = [i for i in range(n) if i not in idx]
            return pooled[list(idx)].mean() - pooled[rest].mean()

        obs = _stat(tuple(range(nA)))
        null = np.array([_stat(c) for c in combinations(range(n), nA)])
        return float(obs), float((np.abs(null) >= abs(obs) - 1e-12).mean()), len(null)

    print(f"{'pair':>18} {'obs diff':>9} {'p':>8} {'n_perm':>7} {'p_min':>7}")
    perm_rows = []
    for _a in range(len(ratios)):
        for _b in range(_a + 1, len(ratios)):
            _hi_r, _lo_r = ratios[_b], ratios[_a]
            _Hr, _Lr = acc_seeds[_hi_r], acc_seeds[_lo_r]
            if len(_Hr) < 2 or len(_Lr) < 2:
                print(f"{_hi_r:.3f} vs {_lo_r:.3f}   skipped (시드 부족)")
                continue
            _obs, _p, _n = perm_test(on_grid(_Hr), on_grid(_Lr))
            perm_rows.append(dict(hi=_hi_r, lo=_lo_r, obs=_obs, p=_p, n_perm=_n))
            print(f"{_hi_r:.3f} vs {_lo_r:.3f}   {_obs:9.3f} {_p:8.4f} {_n:7d} {2.0/_n:7.3f}")
    print("\np_min = 2/n_perm. 3+3 이면 0.100 이라 p<0.05 가 원리적으로 불가능.")
    return


@app.cell
def _(os):
    from torch.utils.data import Dataset
    from torchvision import transforms
    from PIL import Image

    class CMNISTDataset(Dataset):
        def __init__(self, root_dir, train=True, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.train = train
            if train:
                self.dataset_size = [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]
            else:
                self.dataset_size = [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]
            self.dataset_type = 10

            if self.transform is None:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                ])

            self.labels = []
            self.img = []
            for t in range(self.dataset_type):
                for item in range(self.dataset_size[t]):
                    self.labels.append(t)
                    image = Image.open(
                        os.path.join(self.root_dir,
                                     f"{'train' if train else 'test'}/{t}/{item}.png")).convert('RGB')
                    self.img.append(image)
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        def __len__(self):
            return len(self.img)

        def __getitem__(self, idx):
            image_original = self.img[idx]
            return self.transform(image_original)

    return (Image,)


@app.cell
def _(Image, np, os):
    def plot_cmnist_grid(root_dir, labels, train=True, seed=18):
        np.random.seed(seed)

        if train:
            dataset_size = [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]
            split = 'train'
        else:
            dataset_size = [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]
            split = 'test'

        img_path = os.path.join(root_dir, f'{split}/0/0.png')
        sample_img = Image.open(img_path)
        img_w, img_h = sample_img.size
        grid = Image.new('RGB', (img_w * 10, img_h * 3))
        for row in range(len(labels)):
            label = labels[row]
            total_images = dataset_size[label]
            matching_end = int(total_images * 0.1)

            matching_indices = list(range(0, matching_end))
            selected_matching = np.random.choice(matching_indices, size=21, replace=False)
            img_path = os.path.join(root_dir, f'{split}/{label}/{selected_matching[0]}.png')
            image = Image.open(img_path).convert('RGB')
            many_color = image.getpixel((0, 0))
            non_matching_by_color = {}
            for idx in range(0, total_images):
                img_path = os.path.join(root_dir, f'{split}/{label}/{idx}.png')
                image = Image.open(img_path).convert('RGB')
                bg_color = image.getpixel((0, 0))

                if bg_color not in non_matching_by_color:
                    non_matching_by_color[bg_color] = []
                non_matching_by_color[bg_color].append((idx, image))

            del non_matching_by_color[many_color]
            colors = list(non_matching_by_color.keys())

            selected_non_matching = []
            for color in colors[:9]:
                imgs = non_matching_by_color[color]
                selected_non_matching.append(imgs[np.random.randint(len(imgs))])

            for col in range(10):
                if col < 7:
                    idx = selected_matching[row * 7 + col]
                    img_path = os.path.join(root_dir, f'{split}/{label}/{idx}.png')
                    image = Image.open(img_path).convert('RGB')
                else:
                    idx = (col - 7)
                    image = selected_non_matching[idx][1]

                grid.paste(image, (col * img_w, row * img_h))

        grid.save(f'figure/cmnist_grid_label.png')
        grid.show()

    root_dir = '/home/dataset/cmnist/0.1'
    plot_cmnist_grid(root_dir, labels=[3, 5, 7], train=True)
    return


@app.cell
def _(Image, plt):
    img1 = Image.open('/home/dataset/cmnist/0.05/train/6/1.png')
    img2 = Image.open('/home/dataset/cmnist/0.15/train/6/1.png')

    _fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.5))
    plt.subplots_adjust(wspace=0.05)

    axes[0].imshow(img1)
    axes[0].axis('off')
    axes[0].set_title('Object ratio : 0.05', y=-0.13)

    axes[1].imshow(img2)
    axes[1].axis('off')
    axes[1].set_title('Object ratio : 0.15', y=-0.13)

    plt.savefig('figure/cmnist_size_compare.pdf', bbox_inches='tight', pad_inches=0)
    plt.gcf()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
