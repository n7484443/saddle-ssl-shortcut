import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    from PIL import Image

    device = torch.device("cuda")
    sns.set_theme(style="whitegrid", font="cmr10", font_scale=1.9)
    plt.rcParams["axes.formatter.use_mathtext"] = True
    colors = sns.color_palette("dark")  # Original source setting.
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["pdf.use14corefonts"] = False
    base = Path(__file__).resolve().parent
    return Image, base, plt


@app.cell
def _(Image, base, plt):
    def plot_cmnist_size_compare():
        img1 = Image.open(base / "data/cmnist/0.05/train/6/1.png")
        img2 = Image.open(base / "data/cmnist/0.15/train/6/1.png")
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.5))
        plt.subplots_adjust(wspace=0.05)
        axes[0].imshow(img1)
        axes[0].axis('off')
        axes[0].set_title('Object ratio : 0.05', y=-0.13)
        axes[1].imshow(img2)
        axes[1].axis('off')
        axes[1].set_title('Object ratio : 0.15', y=-0.13)
        dest = base / "stage/dataset/cmnist_size_compare.pdf"
        dest.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(dest, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return dest

    result = plot_cmnist_size_compare()
    print(result)
    return


if __name__ == "__main__":
    app.run()
