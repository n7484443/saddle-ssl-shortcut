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

Figure 8(c) is linked to the ReLU MLP renderer; `figures/simclr_dln` contains the separate DLN comparison.
The original training cache for that panel is unavailable, so the historical model is not independently confirmed.
Figure 11 uses CPU-initialized results and a theoretical curve with corrected initial values.
Figure 28 uses the selected five-feature result with scaling 3e-4 and learning rate 7.5e-8.
Figure 42 shows the mean over five seeds. Its SD version adds bands only for depths with positive lower bounds at every slope.
