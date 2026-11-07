# Figure code

`FIGURES.tsv` lists plotting and experiment code for Figures 1–48.

- `feature_preference`: extent bias, CMNIST, Waterbirds, ImageNet, and RRC.
- `frequency_shortcut`: amplitude bias and backdoored MNIST.
- `faint_vivid`: BC and BC+SH training code and configurations used in the paper.
- `imagenet_analysis`: ImageNet spectral scaling analysis.
- `figures`: marimo notebooks that redraw saved results, including the latest figure revisions.
- `common.py`: shared functions. `common_vicreg.py` retains the older VICReg variance coefficient and epsilon behavior.

Use Python 3.12 or newer. Dependencies are listed in `requirements.txt`.
Open marimo notebooks with commands such as `marimo edit figures/faint_vivid/plot.py` from this directory.
For Jupyter notebooks, use the notebook's directory as the working directory.

MNIST, Waterbirds, ImageNet, and model checkpoints must be supplied separately.
Set data paths in the corresponding code or configuration. Saved CSV/NPZ results and figure examples are included.
Some older theory figures no longer have their original `data.npy` files.
Their experiment notebooks are included, with differences between current configurations and historical runs retained.
Running a full experiment notebook may start training; use the cached plotting notebooks in `figures` to redraw available results.
