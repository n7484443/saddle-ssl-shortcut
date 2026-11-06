
from pathlib import Path
import sys

src_dir = next(p for p in Path(__file__).resolve().parents
               if (p / "common.py").is_file())
sys.path.insert(0, str(src_dir))

import csv
import copy

import torch.utils.data
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from setproctitle import setproctitle
import hydra
from omegaconf import DictConfig

from common import *


device = torch.device("cuda")


# Condition images saved by the generated manifest.
# The original code assumed four conditions. For the fast/slow generator we need two.
# We therefore infer the number of conditions from cfg.num_conditions or manifest.csv.


def _coerce_num_conditions(value):
    """Accept num_conditions as an int, a list of ids, or a comma-separated string."""
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
        if stripped:
            return len([x for x in stripped.split(",") if x.strip()])
    raise ValueError(f"Cannot interpret num_conditions={value!r}.")


def _infer_num_conditions_from_manifest(manifest_path):
    """Infer condition count from cond{i}_image columns in manifest.csv."""
    with open(manifest_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

    indices = []
    for name in fieldnames:
        if name.startswith("cond") and name.endswith("_image"):
            middle = name[len("cond"):-len("_image")]
            if middle.isdigit():
                indices.append(int(middle))

    if not indices:
        raise ValueError(f"Could not infer condition columns from manifest: {manifest_path}")

    expected = list(range(max(indices) + 1))
    if sorted(indices) != expected:
        raise ValueError(
            f"Condition columns are not contiguous in {manifest_path}. "
            f"Found {sorted(indices)}, expected {expected}."
        )

    return max(indices) + 1


def _infer_p_values_from_manifest(manifest_path, num_conditions=None):
    """Infer raw p values from the first manifest row."""
    if num_conditions is None:
        num_conditions = _infer_num_conditions_from_manifest(manifest_path)

    with open(manifest_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)

    if row is None:
        raise ValueError(f"manifest.csv has no rows: {manifest_path}")

    p_values = []
    for i in range(int(num_conditions)):
        key = f"cond{i}_p"
        if key not in row:
            raise ValueError(f"manifest.csv is missing {key}: {manifest_path}")
        p_values.append(int(row[key]))

    return p_values


def _get_num_conditions(cfg=None, root_dir=None, split="train"):
    """Get condition count from cfg first, otherwise infer from manifest.csv."""
    if cfg is not None:
        cfg_value = cfg.get("num_conditions", None)
        n = _coerce_num_conditions(cfg_value)
        if n is not None:
            return n

    if root_dir is not None:
        manifest_path = os.path.join(root_dir, split, "manifest.csv")
        if os.path.exists(manifest_path):
            return _infer_num_conditions_from_manifest(manifest_path)

    return None


def _safe_key_for_p(p):
    """Convert p values into stable metric names, e.g. -50 -> m50, 0 -> 0, 100 -> 100."""
    p = int(p)
    return f"m{abs(p)}" if p < 0 else str(p)


def _as_1d_rgb_stats(x):
    return x.reshape(3, -1)


def _get_target_p(cfg):
    """Return cfg.target_p as int, or None if the run should use all conditions."""
    target_p = cfg.get("target_p", None)
    if target_p is None:
        return None
    return int(target_p)


def _num_conditions_for_target(target_p, num_conditions):
    return 1 if target_p is not None else int(num_conditions)



def _iter_pil_images(images_list):
    """Yield PIL images from either all-condition or single-condition storage."""
    for item in images_list:
        if isinstance(item, (list, tuple)):
            for image in item:
                yield image
        else:
            yield item


def compute_rgb_whitening_stats(images_list, eps=1e-5):
    """
    Estimate one global 3x3 RGB covariance from the TRAIN split only.

    Pixels from all training images/conditions are pooled together.  The returned
    ZCA matrix is then reused unchanged for augmented train views, train-noaug,
    and test images.
    """
    to_tensor = transforms.ToTensor()

    sum_rgb = torch.zeros(3, dtype=torch.float64)
    sum_outer = torch.zeros(3, 3, dtype=torch.float64)
    pixel_count = 0

    for image in _iter_pil_images(images_list):
        x = to_tensor(image).to(torch.float64).reshape(3, -1)  # [3, N]
        sum_rgb += x.sum(dim=1)
        sum_outer += x @ x.T
        pixel_count += x.shape[1]

    if pixel_count == 0:
        raise RuntimeError("Cannot compute whitening statistics from an empty training set.")

    mean = sum_rgb / pixel_count
    covariance = sum_outer / pixel_count - torch.outer(mean, mean)
    covariance = 0.5 * (covariance + covariance.T)  # numerical symmetry

    eigvals, eigvecs = torch.linalg.eigh(covariance)
    eigvals_regularized = eigvals.clamp_min(float(eps))

    whitening_matrix = (
        eigvecs
        @ torch.diag(torch.rsqrt(eigvals_regularized))
        @ eigvecs.T
    )

    return mean.float(), covariance.float(), whitening_matrix.float()


class FixedRGBWhitening:
    """
    Apply fixed dataset-level RGB ZCA whitening:
        x_white = W_train @ (x - mean_train)

    mean_train and W_train must be estimated from the training split only.
    """

    def __init__(self, mean, whitening_matrix):
        self.mean = torch.as_tensor(mean, dtype=torch.float32).clone()
        self.whitening_matrix = torch.as_tensor(
            whitening_matrix, dtype=torch.float32
        ).clone()

        if self.mean.shape != (3,):
            raise ValueError(f"Expected RGB mean shape [3], got {tuple(self.mean.shape)}")
        if self.whitening_matrix.shape != (3, 3):
            raise ValueError(
                f"Expected whitening matrix shape [3, 3], "
                f"got {tuple(self.whitening_matrix.shape)}"
            )

    def __call__(self, x):
        if x.ndim != 3 or x.shape[0] != 3:
            raise ValueError(f"Expected tensor [3, H, W], got {tuple(x.shape)}")

        c, h, w = x.shape
        flat = x.reshape(c, -1)

        mean = self.mean.to(device=x.device, dtype=x.dtype).view(3, 1)
        whitening_matrix = self.whitening_matrix.to(
            device=x.device, dtype=x.dtype
        )

        whitened = whitening_matrix @ (flat - mean)
        return whitened.reshape(c, h, w)


class MNISTFour48PDataset(Dataset):
    """
    Dataset for the generated four-image 48x48 MNIST format:

        root_dir/
          train/
            images/m/000000.png
            images/p33/000000.png
            images/p66/000000.png
            images/d/000000.png
            manifest.csv
          test/
            ...

    manifest.csv must contain cond0_image ... cond3_image and matching
    cond{i}_label, cond{i}_p columns.

    SSL training returns two augmented views.
    - target_p is None: each view is shaped [4, 3, H, W].
    - target_p is set: each view is shaped [3, H, W] for that condition only.

    No-aug / eval returns either all conditions or the selected target_p condition.
    The collate function below converts both cases to ordinary [B, 3, H, W] batches.
    """

    def __init__(
        self,
        root_dir,
        cfg,
        train=True,
        transform=None,
        no_aug=False,
        mean=None,
        covariance=None,
        whitening_matrix=None,
        target_p=None,
        num_conditions=None,
    ):
        self.root_dir = root_dir
        self.train = train
        self.no_aug = no_aug
        self.target_p = None if target_p is None else int(target_p)
        self.dataset_type = dataset_type
        self.num_conditions = None if num_conditions is None else int(num_conditions)

        split = "train" if train else "test"
        self.split_dir = os.path.join(self.root_dir, split)
        image_dir = os.path.join(self.split_dir, "images")
        manifest_path = os.path.join(self.split_dir, "manifest.csv")

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Could not find manifest.csv at {manifest_path}.")
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Could not find images directory at {image_dir}.")

        if self.num_conditions is None:
            self.num_conditions = _infer_num_conditions_from_manifest(manifest_path)

        self.img = []
        self.labels = []
        self.p_values = []

        with open(manifest_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            required_columns = []
            for i in range(self.num_conditions):
                required_columns += [f"cond{i}_image", f"cond{i}_label", f"cond{i}_p"]
            missing = [c for c in required_columns if c not in reader.fieldnames]
            if missing:
                raise ValueError(f"manifest.csv is missing required columns: {missing}")

            for row in reader:
                images = []
                labels = []
                p_vals = []

                for i in range(self.num_conditions):
                    image_rel = row[f"cond{i}_image"]
                    if os.path.isabs(image_rel):
                        image_path = image_rel
                    else:
                        image_path = os.path.join(self.split_dir, image_rel)

                    image = Image.open(image_path).convert("RGB")
                    images.append(image)

                    labels.append(int(row[f"cond{i}_label"]))
                    p_vals.append(int(row[f"cond{i}_p"]))

                if self.target_p is None:
                    self.img.append(images)
                    self.labels.append(torch.tensor(labels, dtype=torch.long))
                    self.p_values.append(torch.tensor(p_vals, dtype=torch.long))
                else:
                    matches = [j for j, p in enumerate(p_vals) if int(p) == self.target_p]
                    if len(matches) != 1:
                        raise ValueError(
                            f"Expected exactly one condition with target_p={self.target_p} in manifest row, "
                            f"but found {len(matches)}. Available p values: {p_vals}"
                        )
                    j = matches[0]
                    self.img.append(images[j])
                    self.labels.append(torch.tensor(labels[j], dtype=torch.long))
                    self.p_values.append(torch.tensor(p_vals[j], dtype=torch.long))

        if len(self.img) == 0:
            raise RuntimeError(f"No images found from manifest at {manifest_path}.")

        whitening_eps = float(cfg.get("whitening_eps", 1e-5))

        # Training dataset computes the statistics once. Test receives these exact
        # train statistics through get_four48_p_cmnist_dataloaders().
        if mean is not None and whitening_matrix is not None:
            self.mean = torch.as_tensor(mean, dtype=torch.float32).clone()
            self.covariance = (
                None
                if covariance is None
                else torch.as_tensor(covariance, dtype=torch.float32).clone()
            )
            self.whitening_matrix = torch.as_tensor(
                whitening_matrix, dtype=torch.float32
            ).clone()
        elif mean is None and whitening_matrix is None:
            self.mean, self.covariance, self.whitening_matrix = (
                compute_rgb_whitening_stats(self.img, eps=whitening_eps)
            )
        else:
            raise ValueError(
                "mean and whitening_matrix must either both be provided or both be None."
            )

        self.std = None  # compatibility only; this whitening version does not use channel std.

        print(f"whitening mean: {self.mean}")
        print(f"whitening covariance:\n{self.covariance}")
        print(f"whitening matrix:\n{self.whitening_matrix}")

        whitening = FixedRGBWhitening(
            mean=self.mean,
            whitening_matrix=self.whitening_matrix,
        )

        # No ColorJitter in the whitening experiment.
        # The whitening transform is fixed: both augmented views use the same
        # train-estimated mean/covariance; randomness only comes from crop/affine.
        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
            transforms.RandomAffine(degrees=5, scale=(0.95, 1.05), shear=5),
            transforms.ToTensor(),
            whitening,
        ])
        self.augment_prime = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
            transforms.RandomAffine(degrees=5, scale=(0.95, 1.05), shear=5),
            transforms.ToTensor(),
            whitening,
        ])

        # train-noaug and test also use exactly the same TRAIN whitening statistics.
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            whitening,
        ])

    def _load_mean_std_from_json(self, root_dir):
        """Load normalization mean/std from generated metadata.

        Priority:
        - mean: colors.json["mean_rgb_empirical"] if present, else stats.json["train_mean_rgb"],
          else top-level colors.json["m"] / 255.
        - std: colors.json["std"] if present, else stats.json["train_std_rgb"], else [0.5]*3.

        This is needed for the border-calibrated fast/slow datasets, where the empirical channel
        std is the experimental variable.
        """
        import json

        colors_path = os.path.join(root_dir, "colors.json")
        stats_path = os.path.join(root_dir, "stats.json")

        if not os.path.exists(colors_path):
            raise FileNotFoundError(f"Could not find colors.json at {colors_path}")

        with open(colors_path, "r") as f:
            colors = json.load(f)

        stats = {}
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                stats = json.load(f)

        if "mean_rgb_empirical" in colors:
            mean_values = colors["mean_rgb_empirical"]
        elif "train_mean_rgb" in stats:
            mean_values = stats["train_mean_rgb"]
        else:
            mean_value = float(colors.get("m", 128)) / 255.0
            mean_values = [mean_value, mean_value, mean_value]

        if "std" in colors:
            std_values = colors["std"]
        elif "train_std_rgb" in stats:
            std_values = stats["train_std_rgb"]
        else:
            std_values = [0.5, 0.5, 0.5]

        if len(mean_values) != 3:
            raise ValueError(f"Expected 3 mean values, got {mean_values}")
        if len(std_values) != 3:
            raise ValueError(f"Expected 3 std values, got {std_values}")

        mean = torch.tensor([float(x) for x in mean_values], dtype=torch.float32)
        std = torch.tensor([max(float(x), 1e-6) for x in std_values], dtype=torch.float32)

        return mean, std

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        images = self.img[idx]

        if self.target_p is not None:
            image = images
            if self.no_aug:
                x = self.transform(image)
                return x, self.labels[idx], self.p_values[idx]

            view1 = self.augment(image)
            view2 = self.augment_prime(image)
            return view1, view2

        if self.no_aug:
            xs = torch.stack([self.transform(image) for image in images], dim=0)
            return xs, self.labels[idx], self.p_values[idx]

        view1 = torch.stack([self.augment(image) for image in images], dim=0)
        view2 = torch.stack([self.augment_prime(image) for image in images], dim=0)
        return view1, view2


class NoAugViewDataset(Dataset):
    """No-augmentation view over an already-loaded MNISTFour48PDataset."""

    def __init__(self, base_dataset):
        self.base = base_dataset
        self.mean = base_dataset.mean
        self.std = base_dataset.std
        self.covariance = base_dataset.covariance
        self.whitening_matrix = base_dataset.whitening_matrix
        self.dataset_type = base_dataset.dataset_type
        self.target_p = base_dataset.target_p

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        images = self.base.img[idx]

        if self.target_p is not None:
            x = self.base.transform(images)
            return x, self.base.labels[idx], self.base.p_values[idx]

        xs = torch.stack([self.base.transform(image) for image in images], dim=0)
        return xs, self.base.labels[idx], self.base.p_values[idx]


def _flatten_condition_collate(batch):
    """
    Convert dataset samples to model batches [B, 3, H, W].

    Supports both modes:
    - all-condition mode: item tensors are [4, 3, H, W], so flatten [B0, 4, ...].
    - target_p mode: item tensors are [3, H, W], so keep [B0, ...].
    """
    if len(batch[0]) == 2:
        view1 = torch.stack([item[0] for item in batch], dim=0)
        view2 = torch.stack([item[1] for item in batch], dim=0)

        if view1.dim() == 5:
            view1 = view1.flatten(0, 1)
            view2 = view2.flatten(0, 1)

        perm = torch.randperm(view1.shape[0])
        return view1[perm], view2[perm]

    images = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.stack([item[1] for item in batch], dim=0)
    p_vals = torch.stack([item[2] for item in batch], dim=0)

    if images.dim() == 5:
        images = images.flatten(0, 1)
        labels = labels.flatten(0, 1)
        p_vals = p_vals.flatten(0, 1)

    perm = torch.randperm(images.shape[0])
    return images[perm], labels[perm], p_vals[perm]


def _logical_batch_size(effective_batch_size, num_conditions, target_p=None):
    return max(1, int(effective_batch_size) // _num_conditions_for_target(target_p, num_conditions))


def get_four48_p_cmnist_dataloaders(root_dir, train_batch_size, test_batch_size, cfg, target_p=None, num_conditions=None):
    if num_conditions is None:
        num_conditions = _get_num_conditions(root_dir=root_dir)

    train_dataset = MNISTFour48PDataset(
        cfg=cfg,
        root_dir=root_dir,
        train=True,
        target_p=target_p,
        num_conditions=num_conditions,
    )

    # Compute whitening statistics from TRAIN only and reuse them for TEST.
    train_mean = train_dataset.mean
    train_covariance = train_dataset.covariance
    train_whitening_matrix = train_dataset.whitening_matrix

    test_dataset = MNISTFour48PDataset(
        cfg=cfg,
        root_dir=root_dir,
        train=False,
        no_aug=True,
        mean=train_mean,
        covariance=train_covariance,
        whitening_matrix=train_whitening_matrix,
        target_p=target_p,
        num_conditions=num_conditions,
    )

    train_noaug_dataset = NoAugViewDataset(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=_logical_batch_size(train_batch_size, num_conditions=num_conditions, target_p=target_p),
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        collate_fn=_flatten_condition_collate,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=_logical_batch_size(test_batch_size, num_conditions=num_conditions, target_p=target_p),
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=_flatten_condition_collate,
    )

    train_noaug_loader = DataLoader(
        train_noaug_dataset,
        batch_size=_logical_batch_size(train_batch_size, num_conditions=num_conditions, target_p=target_p),
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        collate_fn=_flatten_condition_collate,
    )
    return train_loader, test_loader, train_noaug_loader


class BarlowTwins(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.network == "ResNet-18":
            self.backbone = torchvision.models.resnet18(zero_init_residual=False, weights=None)
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.backbone.maxpool = nn.Identity()
            self.size_in = self.backbone.fc.in_features
            self.size_out = 1024
            self.backbone.fc = nn.Identity()
        else:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.LazyLinear(512),
            )
            self.size_in = 512
            self.size_out = 512

        # sizes = [self.size_in, self.size_out, self.size_out]
        # layers = []
        # for i in range(len(sizes) - 2):
        #     layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
        #     # layers.append(nn.BatchNorm1d(sizes[i + 1]))
        #     layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        # self.projector = nn.Sequential(*layers)
        self.projector = nn.Identity()

    def forward(self, x, use_projector=True):
        if not use_projector:
            return self.backbone(x)
        z = self.projector(self.backbone(x))
        return z


def import_dataset(cfg, train_batch_size, test_batch_size):
    target_p = _get_target_p(cfg)
    root_dir = cfg.root_dir + "/" + cfg.dir_name
    num_conditions = _get_num_conditions(cfg=cfg, root_dir=root_dir)
    train_loader, test_loader, train_noaug_loader = get_four48_p_cmnist_dataloaders(
        cfg=cfg,
        root_dir=root_dir,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        target_p=target_p,
        num_conditions=num_conditions,
    )
    print(f"num_conditions: {num_conditions}")
    if target_p is not None:
        print(f"target_p: {target_p}")
    print(f"train dataset size: {len(train_loader.dataset)}")
    print(f"test dataset size: {len(test_loader.dataset)}")
    return train_loader, test_loader, train_noaug_loader


def _get_expected_p_values(cfg):
    target_p = _get_target_p(cfg)
    if target_p is not None:
        return [target_p]

    p_values = cfg.get("p_values", None)
    if p_values is not None:
        return [int(p) for p in p_values]

    root_dir = None
    try:
        root_dir = cfg.root_dir + "/" + cfg.dir_name
    except Exception:
        root_dir = None

    if root_dir is not None:
        manifest_path = os.path.join(root_dir, "train", "manifest.csv")
        if os.path.exists(manifest_path):
            num_conditions = _get_num_conditions(cfg=cfg, root_dir=root_dir)
            return _infer_p_values_from_manifest(manifest_path, num_conditions=num_conditions)

    return None


def _map_p_values_to_indices(p_vals, p_value_to_idx):
    """Map raw p values (e.g., 0, 33, 66, 100) to contiguous class indices."""
    mapped = torch.empty_like(p_vals, dtype=torch.long)
    mapped.fill_(-1)
    for p, idx in p_value_to_idx.items():
        mapped[p_vals == int(p)] = int(idx)

    if bool((mapped < 0).any().item()):
        bad_values = torch.unique(p_vals[mapped < 0]).detach().cpu().tolist()
        raise ValueError(f"Found p values not listed in expected_p_values: {bad_values}")
    return mapped


def _cfg_bool(cfg, name, default=False):
    """Read a boolean Hydra/OmegaConf option with a Python fallback."""
    value = cfg.get(name, default)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _percent(correct, total):
    if total <= 0:
        return float("nan")
    return 100.0 * float(correct) / float(total)


def _mean_values(values_by_p):
    if not values_by_p:
        return float("nan")
    return sum(values_by_p.values()) / len(values_by_p)


def _percent_by_p(correct_by_p, total_by_p, expected_p_values):
    return {
        int(p): _percent(correct_by_p[int(p)], total_by_p[int(p)])
        for p in expected_p_values
    }


def _add_by_p_metrics(metrics, prefix, values_by_p, expected_p_values):
    for p in expected_p_values:
        metrics[f"{prefix}_{_safe_key_for_p(p)}"] = values_by_p[int(p)]


def _write_realtime_metrics(file_path, metrics):
    """Write a compact metric dictionary to the realtime logger."""
    for key, value in metrics.items():
        write_realtime(file_path, key, value)


def train_eval(cfg, train, test, train_noaug, epochs, file_path, lambd=None):
    model = BarlowTwins(cfg).to(device)
    size_in = model.size_in
    size_out = model.size_out
    expected_p_values = _get_expected_p_values(cfg)

    scale = float(cfg.scaling)
    cumulative1, cumulative2 = [], []

    train_iter = iter(train)
    for _ in range(6):
        view1, view2 = next(train_iter)
        cumulative1.append(view1.to(device, non_blocking=True))
        cumulative2.append(view2.to(device, non_blocking=True))

    cumulative1 = torch.cat(cumulative1, dim=0)
    cumulative2 = torch.cat(cumulative2, dim=0)

    def evaluate(m, data):
        view1, view2 = data
        eigvals = compute_matrix_eigenvals(m, view1, view2, embeddings=True)
        return eigvals.max()

    weights_init_scale, top_c_eigval = find_sufficiently_small_init(
        model=model,
        eval_fn=evaluate,
        eval_target=scale,
        data=(cumulative1, cumulative2),
        iterations=100,
    )
    print(f"{weights_init_scale}, {top_c_eigval}")
    scale_weights(model, scale=weights_init_scale)
    # apply_residual_scaling(model, cfg.scaling, mode="fixed", verbose=True)

    base_lr = float(cfg.lr)
    if method == "BarlowTwins":
        loss_fn = BTLoss(use_batchnorm=True, in_feature=size_out, lambd=float(lambd)).to(device)
    elif method == "VICReg":
        loss_fn = VICRegLoss().to(device)
    elif method == "LeJEPA":
        loss_fn = LeJEPALoss().to(device)
    else:
        loss_fn = SimCLRLoss(batch_size, temperature=float(cfg.temperature)).to(device)

    if cfg.optimizer == "SGD":
        optimizer_main = torch.optim.SGD(model.parameters(), lr=base_lr, weight_decay=weight_decay, momentum=0.9)
    else:
        optimizer_main = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    # Diagnostic probes.
    # 1) Legacy per-p digit heads: digit decoding inside each condition.
    # 2) Global digit head: one digit classifier shared across all p values.
    # 3) Condition head: predicts p/color condition directly.
    #
    # The global digit and condition probes are the critical diagnostics for
    # separating "the model learned the color/amplitude condition" from
    # "the model learned digit shape under that condition".
    linear_heads_by_p = nn.ModuleDict({
        _safe_key_for_p(p): nn.Linear(size_in, dataset_type).to(device)
        for p in expected_p_values
    })
    head_digit_global = nn.Linear(size_in, dataset_type).to(device)
    head_condition = None
    if len(expected_p_values) > 1:
        head_condition = nn.Linear(size_in, len(expected_p_values)).to(device)

    state_dict_heads_by_p = {
        key: copy.deepcopy(head.state_dict())
        for key, head in linear_heads_by_p.items()
    }
    state_dict_digit_global = copy.deepcopy(head_digit_global.state_dict())
    state_dict_condition = None if head_condition is None else copy.deepcopy(head_condition.state_dict())
    p_value_to_idx = {int(p): i for i, p in enumerate(expected_p_values)}

    bar = tqdm(range(epochs), mininterval=10, maxinterval=10)
    def update_accuracy_lin(current_steps):
        """Train linear probes on frozen features and write diagnostic accuracies.

        Metrics written:
        - accuracy_digit_global: one shared digit head across all p values.
        - accuracy_digit_global_p_<p>: same shared digit head, evaluated within each p.
        - accuracy_condition_probe: p/color-condition prediction accuracy.
        - accuracy_condition_probe_p_<p>: condition-probe recall for each p.
        - accuracy_p_<p>: legacy per-p digit head accuracy.
        - accuracy_macro_p: macro average of the legacy per-p digit heads.
        """
        model.eval()

        all_features_train = []
        all_labels_train = []
        all_p_train = []
        all_features_test = []
        all_labels_test = []
        all_p_test = []

        with torch.inference_mode():
            for image, labels, p_vals in train_noaug:
                image = image.to(device, non_blocking=True)
                feat = model(image, use_projector=False)
                all_features_train.append(feat.detach())
                all_labels_train.append(labels.to(device, non_blocking=True))
                all_p_train.append(p_vals.to(device, non_blocking=True))

            for image, labels, p_vals in test:
                image = image.to(device, non_blocking=True)
                feat = model(image, use_projector=False)
                all_features_test.append(feat.detach())
                all_labels_test.append(labels.to(device, non_blocking=True))
                all_p_test.append(p_vals.to(device, non_blocking=True))

        cached_features_train = torch.cat(all_features_train, dim=0)
        cached_labels_train = torch.cat(all_labels_train, dim=0)  # [N]
        cached_p_train = torch.cat(all_p_train, dim=0)            # [N]

        cached_features_test = torch.cat(all_features_test, dim=0)
        cached_labels_test = torch.cat(all_labels_test, dim=0)    # [N]
        cached_p_test = torch.cat(all_p_test, dim=0)              # [N]

        del all_features_train, all_labels_train, all_p_train
        del all_features_test, all_labels_test, all_p_test

        n_train = cached_features_train.shape[0]
        lp_batch_size = train_noaug.batch_size * _num_conditions_for_target(_get_target_p(cfg), train_noaug.dataset.base.num_conditions)

        # Reset all probes to the same initialization at every checkpoint, so the
        # reported curves reflect encoder quality rather than accumulated probe training.
        for key, head in linear_heads_by_p.items():
            head.load_state_dict(state_dict_heads_by_p[key])
            head.train()
        head_digit_global.load_state_dict(state_dict_digit_global)
        head_digit_global.train()
        if head_condition is not None:
            head_condition.load_state_dict(state_dict_condition)
            head_condition.train()

        probe_params = list(linear_heads_by_p.parameters()) + list(head_digit_global.parameters())
        if head_condition is not None:
            probe_params += list(head_condition.parameters())

        probe_lr = float(cfg.get("probe_lr", 1e-3))
        optimizer_classifier = torch.optim.Adam(probe_params, lr=probe_lr)

        g = torch.Generator(device=device)
        g.manual_seed(0)
        fixed_perm = torch.randperm(n_train, generator=g, device=device)

        lp_epochs = int(cfg.get("probe_epochs", 35))
        lp_bar = tqdm(range(lp_epochs), mininterval=10, maxinterval=10)

        for _ in lp_bar:
            sum_loss_digit_global = 0.0
            total_digit_global_for_loss = 0
            sum_loss_condition = 0.0
            total_condition_for_loss = 0
            sum_losses_by_p = {int(p): 0.0 for p in expected_p_values}
            total_by_p_for_loss = {int(p): 0 for p in expected_p_values}

            for i in range(0, n_train, lp_batch_size):
                idx = fixed_perm[i:i + lp_batch_size]

                feat = cached_features_train[idx]
                labels = cached_labels_train[idx]      # [B]
                p_vals = cached_p_train[idx]           # [B]

                optimizer_classifier.zero_grad(set_to_none=True)

                losses = []

                # Shared digit probe over all p conditions.
                outputs_digit_global = head_digit_global(feat)
                loss_digit_global = F.cross_entropy(outputs_digit_global, labels)
                losses.append(loss_digit_global)
                n_batch = labels.numel()
                sum_loss_digit_global += n_batch * loss_digit_global.item()
                total_digit_global_for_loss += n_batch

                # p/color-condition probe.
                if head_condition is not None:
                    p_targets = _map_p_values_to_indices(p_vals, p_value_to_idx)
                    outputs_condition = head_condition(feat)
                    loss_condition = F.cross_entropy(outputs_condition, p_targets)
                    losses.append(loss_condition)
                    sum_loss_condition += n_batch * loss_condition.item()
                    total_condition_for_loss += n_batch

                # Legacy: independent digit head for each p condition.
                for p in expected_p_values:
                    p_int = int(p)
                    key = _safe_key_for_p(p_int)
                    head = linear_heads_by_p[key]

                    mask = p_vals == p_int
                    if not mask.any():
                        continue

                    outputs_p = head(feat[mask])
                    labels_p = labels[mask]
                    loss_p = F.cross_entropy(outputs_p, labels_p)
                    losses.append(loss_p)

                    n_p = labels_p.numel()
                    sum_losses_by_p[p_int] += n_p * loss_p.item()
                    total_by_p_for_loss[p_int] += n_p

                loss = torch.stack(losses).sum()
                loss.backward()
                optimizer_classifier.step()

            status_parts = [
                f"digit_global: {sum_loss_digit_global / max(total_digit_global_for_loss, 1):.4f}",
            ]
            if head_condition is not None:
                status_parts.append(
                    f"condition: {sum_loss_condition / max(total_condition_for_loss, 1):.4f}"
                )
            status_parts.extend([
                f"p={p:+d}: {sum_losses_by_p[int(p)] / max(total_by_p_for_loss[int(p)], 1):.4f}"
                for p in expected_p_values
            ])
            lp_bar.set_description("loss linear " + ", ".join(status_parts), refresh=False)

        for head in linear_heads_by_p.values():
            head.eval()
        head_digit_global.eval()
        if head_condition is not None:
            head_condition.eval()

        # Legacy per-p digit-head accuracies.
        correct_by_p = {int(p): 0 for p in expected_p_values}
        total_by_p = {int(p): 0 for p in expected_p_values}

        # Shared global digit-head accuracies.
        correct_digit_global = 0
        total_digit_global = 0
        correct_digit_global_by_p = {int(p): 0 for p in expected_p_values}
        total_digit_global_by_p = {int(p): 0 for p in expected_p_values}

        # Condition-probe accuracies.
        correct_condition = 0
        total_condition = 0
        correct_condition_by_p = {int(p): 0 for p in expected_p_values}
        total_condition_by_p = {int(p): 0 for p in expected_p_values}

        with torch.inference_mode():
            # Shared digit head on all test examples.
            pred_digit_global = head_digit_global(cached_features_test).argmax(dim=1)
            correct_digit = pred_digit_global == cached_labels_test
            correct_digit_global += correct_digit.sum().item()
            total_digit_global += cached_labels_test.numel()

            for p in expected_p_values:
                p_int = int(p)
                mask = cached_p_test == p_int
                if not mask.any():
                    continue

                # Same shared digit head, but evaluated condition-wise.
                correct_digit_global_by_p[p_int] += correct_digit[mask].sum().item()
                total_digit_global_by_p[p_int] += mask.sum().item()

                # Legacy p-specific digit head.
                key = _safe_key_for_p(p_int)
                head = linear_heads_by_p[key]
                pred_p_digit = head(cached_features_test[mask]).argmax(dim=1)
                labels_p = cached_labels_test[mask]
                correct_p_digit = pred_p_digit == labels_p
                correct_by_p[p_int] += correct_p_digit.sum().item()
                total_by_p[p_int] += mask.sum().item()

            if head_condition is not None:
                condition_targets = _map_p_values_to_indices(cached_p_test, p_value_to_idx)
                pred_condition = head_condition(cached_features_test).argmax(dim=1)
                correct_cond = pred_condition == condition_targets
                correct_condition += correct_cond.sum().item()
                total_condition += condition_targets.numel()

                for p in expected_p_values:
                    p_int = int(p)
                    mask = cached_p_test == p_int
                    if not mask.any():
                        continue
                    correct_condition_by_p[p_int] += correct_cond[mask].sum().item()
                    total_condition_by_p[p_int] += mask.sum().item()

        acc_by_p = _percent_by_p(correct_by_p, total_by_p, expected_p_values)
        acc_macro_p = _mean_values(acc_by_p)

        acc_digit_global = _percent(correct_digit_global, total_digit_global)
        acc_digit_global_by_p = _percent_by_p(
            correct_digit_global_by_p, total_digit_global_by_p, expected_p_values
        )
        acc_digit_global_macro_p = _mean_values(acc_digit_global_by_p)

        if head_condition is not None:
            acc_condition = _percent(correct_condition, total_condition)
            acc_condition_by_p = _percent_by_p(
                correct_condition_by_p, total_condition_by_p, expected_p_values
            )
        else:
            acc_condition = float("nan")
            acc_condition_by_p = {int(p): float("nan") for p in expected_p_values}

        del cached_features_train, cached_labels_train, cached_p_train
        del cached_features_test, cached_labels_test, cached_p_test

        # Compact logging. By default we only record the metrics needed for the
        # amplitude experiment. Extra overlapping metrics can be enabled from cfg.
        metrics = {
            "accuracy_digit_global": acc_digit_global,
        }

        record_digit_global_by_p = _cfg_bool(cfg, "record_digit_global_by_p", len(expected_p_values) > 1)
        record_condition_probe = _cfg_bool(cfg, "record_condition_probe", head_condition is not None)
        record_condition_probe_by_p = _cfg_bool(cfg, "record_condition_probe_by_p", False)
        record_legacy_per_p = _cfg_bool(cfg, "record_legacy_per_p", False)

        if record_digit_global_by_p:
            metrics["accuracy_digit_global_macro_p"] = acc_digit_global_macro_p
            _add_by_p_metrics(metrics, "accuracy_digit_global_p", acc_digit_global_by_p, expected_p_values)

        if record_condition_probe and head_condition is not None:
            metrics["accuracy_condition_probe"] = acc_condition
            if record_condition_probe_by_p:
                _add_by_p_metrics(metrics, "accuracy_condition_probe_p", acc_condition_by_p, expected_p_values)

        # Legacy metrics are intentionally off by default because they overlap with
        # accuracy_digit_global_p_<p> and can make plots/files hard to read.
        if record_legacy_per_p:
            metrics["accuracy_macro_p"] = acc_macro_p
            _add_by_p_metrics(metrics, "accuracy_p", acc_by_p, expected_p_values)

        metrics["steps"] = current_steps
        _write_realtime_metrics(file_path, metrics)

        model.train()
        return {
            "legacy_digit_by_p": acc_by_p,
            "digit_global": acc_digit_global,
            "digit_global_by_p": acc_digit_global_by_p,
            "condition_probe": acc_condition,
            "condition_probe_by_p": acc_condition_by_p,
        }

    update_accuracy_lin(current_steps=0)
    step = 1
    model.train()
    for _ in bar:
        losses = []
        for image1, image2 in train:
            image1 = image1.to(device, non_blocking=True)
            image2 = image2.to(device, non_blocking=True)
            outputs = model(torch.cat([image1, image2], dim=0))
            z1, z2 = outputs.chunk(2, dim=0)
            _, loss = loss_fn(z1, z2)
            optimizer_main.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer_main.step()

            if step % cfg.report_frequency == 0:
                update_accuracy_lin(current_steps=step)
            step += 1
        write_realtime_list(file_path, "loss", losses)


@hydra.main(version_base=None, config_path="configs", config_name="config_fast_slow_whitening")
def main(cfg: DictConfig):
    global method, network, dataset_type, input_size, weight_decay, batch_size

    method = cfg.method
    network = cfg.network
    dataset_type = 10
    input_size = cfg.input_size

    lambdas = cfg.lambdas
    epochs = cfg.epochs

    weight_decay = float(cfg.weight_decay)
    batch_size = int(cfg.batch_size)

    setproctitle("mnist_four48_p_probe_diagnostics_whitening")
    print(f"lambda: {lambdas}")
    print(f"RGB whitening: enabled (train statistics only, eps={float(cfg.get('whitening_eps', 1e-5))})")
    seed = int(cfg.seed)
    set_seed(seed)
    train, test, train_noaug = import_dataset(cfg=cfg, train_batch_size=batch_size, test_batch_size=2048)
    set_seed(seed)
    target_p = _get_target_p(cfg)
    target_suffix = "all_p" if target_p is None else f"p{_safe_key_for_p(target_p)}"

    train_eval(
        cfg,
        train,
        test,
        train_noaug,
        epochs,
        f"results/{method}_WHITEN_{seed}/lambda_{lambdas}_lr_{cfg.lr:.1e}_{cfg.dir_name}_{target_suffix}",
        lambd=lambdas,
    )


if __name__ == "__main__":
    main()
