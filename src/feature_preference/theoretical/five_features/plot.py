import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter
    import seaborn as sns
    device = torch.device("cuda")
    sns.set_theme(style="whitegrid", font="cmr10", font_scale=1.9)
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams.update({"mathtext.fontset": "cm", "pdf.fonttype": 42, "ps.fonttype": 42})
    colors = sns.color_palette()
    root = Path(__file__).resolve().parent
    return LogLocator, NullFormatter, colors, np, pd, plt, root


@app.cell
def _(colors, np, plt, root):
    def make_single_pair(trial, first=3):
        with np.load(root / trial['key'] / 'run.npz',allow_pickle=False) as cache:
            eigs=cache['eiglist'].copy()
        fig,ax=plt.subplots(figsize=(9,6.6))
        fig.subplots_adjust(left=.18,right=.97,bottom=.17,top=.89)
        for color_index,rank in enumerate((first,first+1)):
            ax.plot(np.arange(400),eigs[rank],color=colors[color_index],lw=1.8,label=rf'$\lambda_{{{rank+1}}}$')
        ax.set(xlabel='Step',ylabel='Eigenvalue',xlim=(0,400))
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
        ax.legend(loc='lower right')
        return fig
    return make_single_pair,


@app.cell
def _(make_single_pair, root):
    figure = make_single_pair({"key": "."}, first=3)
    figure.savefig(root / "simclr_5feature.pdf", bbox_inches="tight", pad_inches=.2)
    figure
    return figure,


if __name__ == "__main__":
    app.run()
