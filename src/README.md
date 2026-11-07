# Source code

[`FIGURES.tsv`](FIGURES.tsv) maps paper figures to plotting and experiment code.

- `feature_preference`: extent bias, CMNIST, Waterbirds, ImageNet, and RRC.
- `frequency_shortcut`: amplitude bias and backdoored MNIST.
- `faint_vivid`: BC and BC+SH training code and configurations.
- `imagenet_analysis`: ImageNet spectral scaling analysis.
- `figures`: marimo notebooks for plotting saved results.
- `common.py` and `common_vicreg.py`: shared experiment utilities.

Dependencies are listed in [`requirements.txt`](requirements.txt).
Run marimo notebooks from this directory:

```console
marimo edit figures/faint_vivid/plot.py
```

For Jupyter notebooks, use the notebook's directory as the working directory.
Experiment notebooks run training; plotting notebooks in `figures` use saved results.

MNIST, Waterbirds, ImageNet, and model checkpoints must be supplied separately.
Set their paths in the corresponding code or configuration.
Saved CSV/NPZ results and dataset generation notebooks are included. Some experiment notebooks require additional result files that are not included.
