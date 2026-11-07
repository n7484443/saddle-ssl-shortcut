import numpy as np
import pandas as pd


def aggregate_original_heatmap(ratios, classes, indices, predictions, steps):
    # Equivalent to ICLR cell15 / NeurIPS cell17. All 50k classes have >=10
    keep = ratios != 0.0
    frames = []
    for step in steps:
        diff = (predictions[step]["bg_top5"].astype(np.int64)
                - predictions[step]["obj_top5"].astype(np.int64)) * 100
        frames.append(pd.DataFrame({
            "Step": int(step.split("_")[1]), "Class": classes[keep],
            "Sample_Idx": indices[keep], "Diff_Top5": diff[keep],
            "BBox Ratio": ratios[keep],
        }))
    df = pd.concat(frames, ignore_index=True)
    df["BBox Bin"] = pd.cut(df["BBox Ratio"], bins=25, labels=False)
    bin_edges = pd.cut(df["BBox Ratio"], bins=25, retbins=True)[1]
    pivot = df.pivot_table(values="Diff_Top5", index="BBox Bin", columns="Step", aggfunc="mean")
    counts = df.pivot_table(values="Diff_Top5", index="BBox Bin", columns="Step", aggfunc="count")
    step_values = sorted(pivot.columns.values)
    pivot = pivot[step_values]
    counts = counts[step_values]
    return df, pivot, counts, np.concatenate([[0], step_values]), bin_edges
