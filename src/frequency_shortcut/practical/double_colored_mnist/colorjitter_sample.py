import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# C-MNIST ColorJitter viewer""")
    return


@app.cell
def _():
    import io
    import base64
    from pathlib import Path

    from PIL import Image
    from torchvision import transforms

    import marimo as mo

    # Generator output directory. Adjust this path for your environment.
    output_dir = "/home/dataset/cmnist_border_fast_slow_only/fastslow_m128_seed0"
    split = "train"
    # (condition_id, short name for output filenames)
    conditions = [("fast_r10", "fast"), ("slow_g45", "slow")]
    n_per_condition = 20

    # Output directory
    figure_dir = Path("figure")
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Standard SSL ColorJitter
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)

    def load_paths(cid):
        return sorted((Path(output_dir) / split / "images" / cid).glob("*.png"))[:n_per_condition]

    return (
        Image,
        base64,
        color_jitter,
        conditions,
        figure_dir,
        io,
        load_paths,
        mo,
    )


@app.cell
def _(Image, color_jitter, conditions, figure_dir, load_paths):
    # Save no_jitter and jitter versions of each image in the figure directory.
    def save_all():
        saved = []
        for cid, short in conditions:
            for idx, p in enumerate(load_paths(cid)):
                orig = Image.open(p).convert("RGB")
                aug = color_jitter(orig)
                orig_path = figure_dir / f"{idx}_{short}_no_jitter.png"
                aug_path = figure_dir / f"{idx}_{short}_jitter.png"
                orig.save(orig_path)
                aug.save(aug_path)
                saved += [orig_path, aug_path]
        return saved

    saved_files = save_all()
    print(f"saved {len(saved_files)} files to {figure_dir.resolve()}")
    return


@app.cell
def _(Image, base64, color_jitter, conditions, io, load_paths, mo):
    # Preview, separate from the saved output
    def to_img(pil, scale=3):
        pil = pil.resize((pil.width * scale, pil.height * scale), Image.NEAREST)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return mo.image(src=f"data:image/png;base64,{b64}")

    def preview():
        blocks = []
        for cid, short in conditions:
            tiles = []
            for p in load_paths(cid):
                orig = Image.open(p).convert("RGB")
                aug = color_jitter(orig)
                tiles.append(mo.hstack([to_img(orig), to_img(aug)], gap=0.2, align="center"))
            blocks.append(
                mo.vstack([
                    mo.md(f"### {cid} ({len(tiles)})  ·  left=no_jitter, right=jitter"),
                    mo.hstack(tiles, wrap=True, gap=0.5),
                ])
            )
        return mo.vstack(blocks)

    preview()
    return


if __name__ == "__main__":
    app.run()