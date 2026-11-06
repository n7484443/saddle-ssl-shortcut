from pathlib import Path
import sys

src_dir = next(p for p in Path(__file__).resolve().parents
               if (p / "common.py").is_file() and (p / "feature_preference").is_dir())
sys.path.insert(0, str(src_dir))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomResizedCrop
import numpy as np
import polars as pl
from tqdm import tqdm
from scipy.linalg import hadamard
import random
import os
import math
import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
import common

device = torch.device("cuda")

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
IMG_SIZE = 28


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def as_list(value, cast, none_token=None):
    """Normalize a config entry into a flat python list.

    Accepts scalars, lists, ListConfig, nested lists, and comma separated
    strings (e.g. "None,0.7", which is how unquoted YAML values arrive).
    If none_token is given, a None value maps to it instead of being dropped.
    """
    if value is None:
        return [] if none_token is None else [none_token]
    if isinstance(value, ListConfig):
        value = OmegaConf.to_object(value)
    if isinstance(value, str) and "," in value:
        value = [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, (list, tuple, set)):
        out = []
        for v in value:
            out.extend(as_list(v, cast, none_token))
        return out
    return [cast(value)]


def parse_spuriosities(cfg):
    """Return spuriosity values as the string tokens used in the .pt filenames."""
    raw = cfg.get("spuriosities", None)
    if raw is None:
        raw = cfg.get("spuriosity", None)
    if raw is None:
        return ["None"]
    tokens = as_list(raw, str, none_token="None")
    return tokens if tokens else ["None"]


def spur_to_float(token):
    """'None' / 'null' -> None, otherwise float."""
    if token is None or str(token).lower() in ("none", "null", "nan", ""):
        return None
    try:
        return float(token)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Trigger construction
# ---------------------------------------------------------------------------

def _expand(base, size_mult, expand):
    """(C, L) -> (C, L * size_mult).

    tile   : [a b c d] -> [a b c d a b c d]  (repeat the whole pattern)
    repeat : [a b c d] -> [a a b b c c d d]  (stretch each element)
    """
    if size_mult == 1:
        return base
    if expand == "tile":
        return base.repeat(1, size_mult)
    if expand == "repeat":
        return base.repeat_interleave(size_mult, dim=1)
    raise ValueError(f"unknown trigger_expand: {expand}")


def get_hadamard_triggers(n_classes=10, intensity=1.0, size_mult=1, expand="tile"):
    """Hadamard triggers, scaled up by size_mult along the flattened image.

    size_mult=1 -> len 16 : 1x16
    size_mult=2 -> len 32 : 1x28 + 1x4
    size_mult=3 -> len 48 : 1x28 + 1x20
    size_mult=4 -> len 64 : 2x28 + 1x8
    """
    H = hadamard(16)
    # row 0 is all-ones, so skip it and use rows 1..n_classes
    base = torch.tensor(H[1:n_classes + 1], dtype=torch.float32)
    return _expand(base, size_mult, expand) * intensity


def get_onehot_triggers(n_classes=10, intensity=1.0, size_mult=1, expand="tile"):
    """Baseline trigger: +1 at the backdoor class, -1/(C-1) elsewhere."""
    base = torch.full((n_classes, n_classes), -1.0 / (n_classes - 1))
    idx = torch.arange(n_classes)
    base[idx, idx] = 1.0
    return _expand(base, size_mult, expand) * intensity


def build_triggers(trigger_type, intensity, size_mult, expand, n_classes=10):
    if trigger_type == "hadamard":
        return get_hadamard_triggers(n_classes, intensity, size_mult, expand)
    return get_onehot_triggers(n_classes, intensity, size_mult, expand)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_MNIST_CACHE = {}


def load_mnist_base(train=True):
    """Load and normalize MNIST once, then hand out copies (grid rebuilds often)."""
    key = bool(train)
    if key not in _MNIST_CACHE:
        mnist = torchvision.datasets.MNIST(root='/home/dataset/mnist', train=train, download=False)
        images = mnist.data.float().unsqueeze(1) / 255.0
        images = (images - MNIST_MEAN[0]) / MNIST_STD[0]
        _MNIST_CACHE[key] = (images.contiguous(), mnist.targets.clone())
    images, targets = _MNIST_CACHE[key]
    return images.clone(), targets.clone()


class BackdoorMNIST(Dataset):
    def __init__(
        self,
        bd_path,
        cfg,
        train=True,
        mode='ssl',
        noise_std=0.1,
        subset_indices=None,
        rrc_scale=(0.95, 1.0),
        rrc_ratio=(0.95, 1.05),
        intensity=None,
        size_mult=None
    ):
        images, targets = load_mnist_base(train)
        bd_classes = torch.load(bd_path, weights_only=True).long()

        # cfg.intensity / cfg.size_mult may be lists;
        # fall back to the first value.
        self.intensity = (
            float(intensity)
            if intensity is not None
            else as_list(cfg.intensity, float)[0]
        )

        self.size_mult = (
            int(size_mult)
            if size_mult is not None
            else as_list(cfg.get('size_mult', 1), int)[0]
        )

        self.expand = cfg.get('trigger_expand', 'tile')
        self.mode = mode
        self.noise_std = noise_std
        self.img_size = IMG_SIZE
        self.rrc_scale = rrc_scale
        self.rrc_ratio = rrc_ratio

        # ---------------------------------------------------------
        # Apply subset
        # ---------------------------------------------------------
        if subset_indices is not None:
            images = images[subset_indices]
            targets = targets[subset_indices]
            bd_classes = bd_classes[subset_indices]

        # ---------------------------------------------------------
        # Build trigger
        # ---------------------------------------------------------
        triggers = build_triggers(
            cfg.trigger_type,
            self.intensity,
            self.size_mult,
            self.expand
        )

        self.trigger_len = triggers.shape[1]

        assert self.trigger_len <= IMG_SIZE * IMG_SIZE, \
            "trigger is larger than the image"

        self.trigger_rows = math.ceil(
            self.trigger_len / IMG_SIZE
        )
        images = images.contiguous()
        flat_images = images.view(images.size(0), -1)

        trigger_values = triggers[bd_classes]

        if self.size_mult <= 4:
            flat_images[:, :self.trigger_len] = trigger_values

        else:
            # Keep the first 64 trigger values at the top.
            top_len = 64

            flat_images[:, :top_len] = \
                trigger_values[:, :top_len]

            # Put only the additional trigger values at the bottom.
            bottom_len = self.trigger_len - top_len

            flat_images[:, -bottom_len:] = \
                trigger_values[:, top_len:]

        # ---------------------------------------------------------
        # Store tensors
        # ---------------------------------------------------------
        self.images = images.to(device)
        self.labels = targets.to(device)
        self.bd_classes = bd_classes

    def _apply_rrc(self, img):
        """Apply RandomResizedCrop directly on the GPU tensor."""
        i, j, h, w = RandomResizedCrop.get_params(
            img,
            scale=self.rrc_scale,
            ratio=self.rrc_ratio
        )

        return TF.resized_crop(
            img,
            i,
            j,
            h,
            w,
            [self.img_size, self.img_size]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if self.mode == 'ssl':
            view1 = self._apply_rrc(img)
            view2 = self._apply_rrc(img)

            if self.noise_std > 0:
                view1 = (
                    view1
                    + torch.randn_like(view1) * self.noise_std
                )

                view2 = (
                    view2
                    + torch.randn_like(view2) * self.noise_std
                )

            return view1, view2, label

        else:
            return img, label


# ---------------------------------------------------------------------------
# Model / loss
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, hidden_dims=[256, 256, 256, 256], proj_dim=32):
        super().__init__()
        layers = []
        in_dim = 28 * 28
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        self.feature_dim = hidden_dims[-1]

        self.projector = nn.Linear(self.feature_dim, proj_dim)

    def forward(self, x, use_projector=True):
        x = x.view(x.size(0), -1)
        features = self.backbone(x)
        if use_projector:
            return self.projector(features)
        return features


class BarlowTwinsLoss(nn.Module):
    def __init__(self, proj_dim, lambd=0.005):
        super().__init__()
        self.lambd = lambd
        self.bn = nn.BatchNorm1d(proj_dim, affine=False)

    def forward(self, z1, z2):
        batch_size = z1.size(0)
        z1 = self.bn(z1)
        z2 = self.bn(z2)
        c = torch.mm(z1.T, z2) / batch_size

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return c, loss

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def train_linear_probe(model, train_loader, epochs=30, verbose=False):
    """Train a linear probe on the frozen encoder."""
    model.eval()

    classifier = nn.Linear(model.feature_dim, 10).to(device)
    optimizer = torch.optim.Adam(classifier.parameters())

    for epoch in range(epochs):
        classifier.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                features = model(images, use_projector=False)
            logits = classifier(features)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  [probe] Epoch {epoch + 1}/{epochs}: Loss = {total_loss / len(train_loader):.4f}")

    return classifier


def test_with_random_backdoor(model, test_loader, classifier):
    """Test with random backdoor (spuriosity=0)."""
    model.eval()
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            features = model(images, use_projector=False)
            logits = classifier(features)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


# ---------------------------------------------------------------------------
# Result table (polars)
# ---------------------------------------------------------------------------

SCHEMA = {
    "seed": pl.Int64,
    "spuriosity": pl.Utf8,        # raw token, e.g. "None" or "0.7"
    "spuriosity_val": pl.Float64,  # null when the token is "None"
    "subset_ratio": pl.Float64,
    "scaling": pl.Float64,
    "trigger_type": pl.Utf8,
    "trigger_expand": pl.Utf8,
    "size_mult": pl.Int64,
    "trigger_len": pl.Int64,
    "trigger_rows": pl.Int64,
    "intensity": pl.Float64,
    "epoch": pl.Int64,
    "step": pl.Int64,
    "test_acc": pl.Float64,
}


def records_to_df(records):
    return pl.DataFrame(records, schema=SCHEMA)


def _pivot(df, values, index, on):
    """polars renamed the pivot argument from columns= to on=; support both."""
    try:
        return df.pivot(values=values, index=index, on=on, aggregate_function="mean")
    except TypeError:
        return df.pivot(values=values, index=index, columns=on, aggregate_function="mean")


def save_results(records, out_dir):
    """Write the long-format table plus size x intensity pivots per eval point."""
    if not records:
        return None
    os.makedirs(out_dir, exist_ok=True)
    df = records_to_df(records).sort(["spuriosity", "size_mult", "intensity", "step"])

    df.write_parquet(os.path.join(out_dir, "grid_long.parquet"))
    df.write_csv(os.path.join(out_dir, "grid_long.csv"))

    pivot_dir = os.path.join(out_dir, "pivot")
    os.makedirs(pivot_dir, exist_ok=True)
    for (spur, step), sub in df.group_by(["spuriosity", "step"], maintain_order=True):
        wide = _pivot(sub, values="test_acc", index="intensity", on="size_mult").sort("intensity")
        wide.write_csv(os.path.join(pivot_dir, f"spur{spur}_step{step}.csv"))

    # Summary pivot at the last measured step
    last = df.filter(pl.col("step") == pl.col("step").max())
    _pivot(last, values="test_acc", index=["spuriosity", "intensity"], on="size_mult") \
        .sort(["spuriosity", "intensity"]).write_csv(os.path.join(out_dir, "grid_final_pivot.csv"))

    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(train_loader, cfg, train_non_aug_loader, test_loader, meta, records, out_dir):
    max_epoch = cfg.max_epoch

    print(f"\n{'=' * 68}")
    print(f"spuriosity={meta['spuriosity']} | size_mult={meta['size_mult']} "
          f"(len={meta['trigger_len']}, rows={meta['trigger_rows']}) "
          f"| intensity={meta['intensity']:.4f} | epochs={max_epoch}")
    print(f"{'=' * 68}")

    model = Encoder(proj_dim=128).to(device)

    if cfg.scaling != -1:
        cumulative1, cumulative2 = [], []
        for i in range(6):
            view1, view2, _ = next(iter(train_loader))
            cumulative1.append(view1)
            cumulative2.append(view2)
        cumulative1, cumulative2 = torch.cat(cumulative1, dim=0), torch.cat(cumulative2, dim=0)

        def evaluate(m, data):
            view1, view2 = data
            eigvals = common.compute_matrix_eigenvals(m, view1, view2, embeddings=True)
            return eigvals.max()

        weights_init_scale, top_c_eigval = common.find_sufficiently_small_init(
            model=model, eval_fn=evaluate, eval_target=cfg.scaling,
            data=(cumulative1, cumulative2), iterations=100)
        print(f"{weights_init_scale}, {top_c_eigval}")
        common.scale_weights(model, weights_init_scale, scale_bias=True)

    loss_fn = BarlowTwinsLoss(proj_dim=128, lambd=5e-3).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-6)

    steps_per_epoch = len(train_loader)
    total_steps = cfg.max_step if cfg.max_step != -1 else max_epoch * steps_per_epoch
    eval_steps = [i for i in range(0, total_steps, cfg.test_step_freq)]
    done_steps = set()
    print(f"  eval at {len(eval_steps)} steps: {sorted(eval_steps)}")

    def do_eval(epoch, step):
        if step in done_steps:
            return
        done_steps.add(step)
        classifier = train_linear_probe(model, train_non_aug_loader, epochs=cfg.get('probe_epoch', 30))
        test_acc = test_with_random_backdoor(model, test_loader, classifier)
        records.append({**meta, "epoch": epoch, "step": step, "test_acc": test_acc})
        save_results(records, out_dir)  # checkpoint results so a crash loses nothing
        print(f"  [eval] epoch={epoch} step={step} acc={test_acc:.2f}%")
        model.train()

    step = 0
    for epoch in tqdm(range(max_epoch)):
        model.train()
        total_loss = 0
        for view1, view2, _ in train_loader:
            view1, view2 = view1.to(device), view2.to(device)
            z1, z2 = model(view1), model(view2)
            _, loss = loss_fn(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            step += 1

            if step in eval_steps:
                do_eval(epoch + 1, step)
            if step == cfg.max_step:
                do_eval(epoch + 1, step)
                return model, step

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Loss = {total_loss / len(train_loader):.4f}")
        if (epoch + 1) in as_list(cfg.get('test_epoch', []), int):
            do_eval(epoch + 1, step)

    return model, step


def bd_paths(directory, spur):
    train_path = os.path.join(directory, f"train_spur{spur}_seed42.pt")
    test_path = os.path.join(directory, f"test_spur{spur}_seed42.pt")
    return train_path, test_path


def check_bd_files(directory, spur_tokens):
    """Fail fast if any backdoor label file is missing, before burning GPU hours."""
    missing = []
    for spur in spur_tokens:
        for path in bd_paths(directory, spur):
            if not os.path.exists(path):
                missing.append(path)
    if missing:
        available = sorted(os.listdir(directory)) if os.path.isdir(directory) else []
        raise FileNotFoundError(
            "Missing backdoor label files:\n  " + "\n  ".join(missing) +
            "\nFiles present in the directory:\n  " + "\n  ".join(available)
        )


def run_one_setting(cfg, spur, intensity, size_mult, indices, records, out_dir, directory):
    """Run a single grid cell (spuriosity, size_mult, intensity)."""
    set_seed(cfg.seed)  # every cell starts from the same init / shuffle order

    train_path, test_path = bd_paths(directory, spur)

    train_dataset = BackdoorMNIST(bd_path=train_path, train=True, mode='ssl',
                                  subset_indices=indices, cfg=cfg,
                                  intensity=intensity, size_mult=size_mult)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    train_linear_dataset = BackdoorMNIST(bd_path=train_path, train=True, mode='linear',
                                         subset_indices=indices, cfg=cfg,
                                         intensity=intensity, size_mult=size_mult)
    train_linear_loader = DataLoader(train_linear_dataset, batch_size=128, shuffle=True, num_workers=0)

    test_dataset = BackdoorMNIST(bd_path=test_path, train=False, mode='eval', cfg=cfg,
                                 intensity=intensity, size_mult=size_mult)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0)

    meta = {
        "seed": int(cfg.seed),
        "spuriosity": str(spur),
        "spuriosity_val": spur_to_float(spur),
        "subset_ratio": float(cfg.subset_ratio),
        "scaling": float(cfg.scaling),
        "trigger_type": str(cfg.trigger_type),
        "trigger_expand": str(cfg.get('trigger_expand', 'tile')),
        "size_mult": int(size_mult),
        "trigger_len": int(train_dataset.trigger_len),
        "trigger_rows": int(train_dataset.trigger_rows),
        "intensity": float(intensity),
    }

    model, step = train(train_loader, cfg, train_linear_loader, test_loader, meta, records, out_dir)

    # Final measurement, unless this exact step was already recorded
    already = any(r["step"] == step and r["size_mult"] == size_mult
                  and r["intensity"] == intensity and r["spuriosity"] == str(spur)
                  for r in records)
    if not already:
        classifier = train_linear_probe(model, train_linear_loader, epochs=cfg.get('probe_epoch', 30))
        test_acc = test_with_random_backdoor(model, test_loader, classifier)
        records.append({**meta, "epoch": cfg.max_epoch, "step": step, "test_acc": test_acc})
        print(f"Final test accuracy: {test_acc:.2f}%")

    del train_loader, train_linear_loader, test_loader
    del train_dataset, train_linear_dataset, test_dataset, model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="config_grid")
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    spur_tokens = parse_spuriosities(cfg)

    raw_sizes = cfg.get('size_mults', None)
    if raw_sizes is None:
        raw_sizes = cfg.get('size_mult', [i for i in range(1, 9)])
    size_mults = sorted(set(as_list(raw_sizes, int)))

    raw_intensities = cfg.get('intensities', None)
    if raw_intensities is None:
        raw_intensities = cfg.intensity  # scalar or list, both fine
    intensities = sorted(set(as_list(raw_intensities, float)))

    n_runs = len(spur_tokens) * len(size_mults) * len(intensities)
    print(f"grid: spuriosity={spur_tokens} x size_mults={size_mults} "
          f"x intensities={intensities}  ({n_runs} runs)")

    directory = '/home/dataset/mnist_backdoor/'
    check_bd_files(directory, spur_tokens)

    spur_tag = "-".join(spur_tokens)
    step_tag = cfg.max_step if cfg.max_step != -1 else cfg.max_epoch
    out_dir = (f"result/{spur_tag}/{cfg.seed}_seed/"
               f"grid_{cfg.scaling}_{step_tag}_{cfg.subset_ratio}subset")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    image_count = 60000
    n = int(image_count * cfg.subset_ratio)
    indices = torch.randperm(image_count)[:n]  # the whole grid shares one subset

    records = []
    for spur in spur_tokens:
        for size_mult in size_mults:
            for intensity in intensities:
                run_one_setting(cfg, spur, intensity, size_mult, indices,
                                records, out_dir, directory)
                save_results(records, out_dir)

    df = save_results(records, out_dir)
    print(f"\nSaved {len(records)} rows to {out_dir}")
    if df is not None:
        last = df.filter(pl.col("step") == pl.col("step").max())
        print("\n=== final acc (row=[spuriosity, intensity], col=size_mult) ===")
        print(_pivot(last, values="test_acc", index=["spuriosity", "intensity"], on="size_mult")
              .sort(["spuriosity", "intensity"]))


if __name__ == "__main__":
    main()