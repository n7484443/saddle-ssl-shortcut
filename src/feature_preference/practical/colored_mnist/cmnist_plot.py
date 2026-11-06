import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    # %load_ext autoreload
    # %autoreload 2
    # Use marimo Settings > Runtime > Module autoreload instead.
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
    # ratios = [0.05, 0.1, 0.15]
    lambdas = 0.05
    loss = []
    acc = []
    acc_types = []
    step_list = []
    eig_list = []
    seed = 0
    # directory = f"datas/lambda_{lambdas}_{seed}_scaling9e-2_lr2e-4"
    directory = f"datas/lambda_{lambdas}_{seed}"
    # directory = f"datas/BarlowTwins_{seed}"
    for _r in ratios:
        path = f"{_r}"
        # path = f"/ratio_{_r}_lambda_{lambdas}_lr_7.0e-05"
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
    background_predictability = 70

    fig, axs = plt.subplots(1, 1, figsize=(13, 5.3))
    # fig, axs = plt.subplots(1, 2)
    epochs = len(step_list[0])

    positions = []
    ran = 350
    for r in range(len(ratios)):
        pos = np.where(acc[r] >= background_predictability * 1.005) # 0.5% increase
        if len(pos[0]) > 0:
            positions.append(pos[0][0])
            axs.axvline(step_list[r][positions[-1]], linestyle="--", color=colors[r], linewidth=4, alpha=0.7)
        axs.plot(step_list[r][:ran], acc[r][:ran], alpha=0.6, linewidth=5, color=colors[r], label=f"Object ratio : {ratios[r]:.3f}")

    axs.text(600, 68.5, "background-label correlation")

    # axs.set_xlim(0-1, 5300)
    # axs.set_ylim(69, 71)
    axs.set_ylim(65, 77.5)
    axs.axhline(background_predictability, linestyle="--", color="black", linewidth=2, alpha=0.6)
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
                y = np.clip(y, 0, None)   # baseline 
        return _trapz(y, x)

    CEILING = 100.0 

    areas = {}
    for _i, _ratio in enumerate(ratios):
        _x, _y = step_list[_i][1:ran], acc[_i][1:ran]
        _raw   = curve_area(_x, _y)
        _above = curve_area(_x, _y, baseline=background_predictability)
        _pos   = curve_area(_x, _y, baseline=background_predictability, clip=True)

        _span = _x[-1] - _x[0]
        _max_above = (CEILING - background_predictability) * _span  
        _rel     = _above / _max_above * 100.0
        _rel_pos = _pos   / _max_above
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
                        os.path.join(self.root_dir, f"{'train' if train else 'test'}/{t}/{item}.png")).convert('RGB')
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
    # import os
    # import matplotlib.pyplot as plt
    # import numpy as np

    # import os
    # import numpy as np
    # from PIL import Image
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
