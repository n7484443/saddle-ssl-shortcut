import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    def correct_cmsy10_minus_mapping(raw_pdf, final_pdf):
        """Correct only Cmsy10 CID 00a1 Unicode; keep its glyph and all graphics.

        Legacy CM uses slot 00a1 for the minus glyph. Matplotlib's font42
        ToUnicode assigns that slot U+00A1, although the painted glyph is minus.
        The original Type3 figure extracted U+2212. This explicit pipeline step
        fixes searchable text only, retaining the uncorrected fresh PDF.
        """
        from decimal import Decimal
        from pathlib import Path
        import hashlib
        from pypdf import PdfReader, PdfWriter
        from pypdf.generic import FloatObject, DictionaryObject

        raw_pdf, final_pdf = Path(raw_pdf), Path(final_pdf)
        assert raw_pdf.resolve() != final_pdf.resolve()
        reader = PdfReader(raw_pdf)
        writer = PdfWriter()
        writer.clone_document_from_reader(reader)
        before = b"<00a1> <00a1> [<00a1>]"
        after = b"<00a1> <00a1> [<2212>]"
        changes, matched = [], 0
        for obj in writer._objects:
            if not isinstance(obj, DictionaryObject):
                continue
            if str(obj.get("/BaseFont", "")).split("+")[-1] != "Cmsy10":
                continue
            if obj.get("/Subtype") != "/Type0":
                continue
            matched += 1
            assert obj["/Encoding"] == "/Identity-H"
            cmap = obj["/ToUnicode"]
            data = cmap.get_data()
            if before in data:
                assert data.count(before) == 1 and after not in data
                cmap.set_data(data.replace(before, after))
                changes.append({"font": str(obj["/BaseFont"]),
                                "cid_hex": "00a1", "old_unicode": "U+00A1",
                                "new_unicode": "U+2212"})
            else:
                assert data.count(after) == 1, "Unexpected Cmsy10 CMap"
        assert matched == 1, "Expected exactly one used Cmsy10 Type0 font"

        # pypdf otherwise rounds unchanged decimal dictionary coordinates.
        original_myrepr = FloatObject.myrepr
        def exact_float_repr(value):
            text = format(Decimal(repr(float(value))), "f")
            return text.rstrip("0").rstrip(".") if "." in text else text
        try:
            FloatObject.myrepr = exact_float_repr
            with final_pdf.open("wb") as stream:
                writer.write(stream)
        finally:
            FloatObject.myrepr = original_myrepr
        return {
            "raw_pdf": str(raw_pdf),
            "raw_sha256": hashlib.sha256(raw_pdf.read_bytes()).hexdigest(),
            "final_pdf": str(final_pdf),
            "final_sha256": hashlib.sha256(final_pdf.read_bytes()).hexdigest(),
            "changes": changes,
            "purpose": "New replot pipeline Unicode correction, not graphics or font substitution",
            "font_program_or_glyph_modified": False,
        }
    return (correct_cmsy10_minus_mapping,)


@app.cell
def _():
    from pathlib import Path
    import hashlib
    import io
    import json
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pdfplumber
    import pypdfium2 as pdfium
    from pypdf import PdfReader, PdfWriter, PageObject, Transformation
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import marimo as mo

    device = torch.device("cuda")
    sns.set_theme(style="whitegrid", font="cmr10", font_scale=1.9)
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    # A dynamic embedded-font initial setting makes ReportLab omit its unused
    # Helvetica BT/Tf preamble. Overlay text is drawn only with Matplotlib.
    if "FigureCmr10" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("FigureCmr10", str(
            Path(matplotlib.get_data_path()) / "fonts/ttf/cmr10.ttf",
        )))
    colors = sns.color_palette()
    root = Path(__file__).resolve().parent
    return (
        PageObject, PdfReader, PdfWriter, Transformation, canvas,
        hashlib, io, json, mo, np, pdfium, pdfplumber, plt, root,
    )


@app.cell
def _(hashlib, json, mo, np, pdfplumber, root):
    inputs = {
        "source": root / "spuriosity_orig.pdf",
        "plot_data": root / "heatmap_plot_data.npz",
        "heatmap_record": root / "heatmap_record.json",
        "metadata": root / "crop_metadata.json",
        "high": root / "high_ratio_0p8_crop.png",
        "low": root / "low_ratio_0p2_crop.png",
    }
    mo.stop(not all(path.is_file() for path in inputs.values()),
            mo.md("Waiting for the new full-inference heatmap export."))
    heatmap_record = json.loads(inputs["heatmap_record"].read_text(encoding="utf-8"))
    assert heatmap_record["rows"] == 597804
    assert heatmap_record["unique_samples"] == 49817
    assert heatmap_record["unique_classes"] == 1000
    assert heatmap_record["matrix_shape"] == [25, 12]
    assert heatmap_record["plot_input_from_original_pdf"] is False
    assert heatmap_record["inference_policy"]["smoke"] is False
    expected_base_sha = next(item["sha256"] for item in heatmap_record["exports"]
                             if item["path"].lower().endswith(".pdf"))
    with np.load(inputs["plot_data"], allow_pickle=False) as heatmap_data:
        ratio_bin_edges = heatmap_data["y_edges"].copy()
    assert ratio_bin_edges.shape == (26,)
    input_hashes = {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in inputs.items()
    }
    crop_metadata = json.loads(inputs["metadata"].read_text(encoding="utf-8"))
    examples = {item["role"]: item for item in crop_metadata["images"]}
    for role in ("high", "low"):
        assert examples[role]["output_file_sha256"] == input_hashes[role]
    with pdfplumber.open(inputs["source"]) as original:
        original_page = original.pages[0]
        cells = [
            item for item in original_page.rects
            if isinstance(item["non_stroking_color"], (tuple, list))
            and len(item["non_stroking_color"]) == 3
        ]
        assert len(cells) == 300
        x_min = min(item["x0"] for item in cells)
        x_max = max(item["x1"] for item in cells)
        y_edges = sorted({round(item[key], 6) for item in cells for key in ("y0", "y1")})
        assert len(y_edges) == 26
    # New PDF cells supply only composition coordinates, not experiment data.
    bin_specs = {}
    for role, expected_bin in (("high", 19), ("low", 4)):
        ratio = examples[role]["paper_fig4_bbox_ratio"]
        bin_index = int(np.searchsorted(ratio_bin_edges, ratio, side="left") - 1)
        assert bin_index == expected_bin
        interval = ratio_bin_edges[bin_index:bin_index+2].tolist()
        assert interval[0] < ratio <= interval[1]
        bin_specs[role] = {
            "index": bin_index, "ratio_interval": interval, "actual_ratio": ratio,
            "source_pdf_rect": [x_min, y_edges[bin_index], x_max, y_edges[bin_index+1]],
        }
    return bin_specs, input_hashes, inputs


@app.cell
def _(
    PageObject, PdfReader, PdfWriter, Transformation, canvas, io, pdfium, plt,
):
    def make_header_overlay(width, height, headers):
        """Match the fresh heatmap's Matplotlib embedded fonttype42 Cmr10."""
        output = io.BytesIO()
        with plt.rc_context({
            "pdf.fonttype": 42, "font.family": "cmr10",
            "font.style": "normal", "font.weight": "normal",
            "text.usetex": False, "savefig.bbox": None,
        }):
            text_figure = plt.figure(
                figsize=(width / 72, height / 72), facecolor="none",
            )
            for x, y, label in headers:
                text_figure.text(
                    x / width, y / height, label,
                    family="cmr10", fontsize=22.8, style="normal",
                    weight="normal", color="0.15", ha="center", va="baseline",
                )
            text_figure.savefig(
                output, format="pdf", transparent=True,
                bbox_inches=None, pad_inches=0,
            )
            plt.close(text_figure)
        output.seek(0)
        return output

    def plot_figure4(source_path, crops, bin_specs, destination):
        """Compose original vectors, bin outlines, and clean crop images."""
        width, height = 1330.0, 752.0
        source_x_offset = 310.0
        source_y_offset = 0.0
        source = PdfReader(source_path).pages[0]
        assert tuple(float(v) for v in source.mediabox) == (0, 0, 1008, 720)
        source_fonts = source["/Resources"].get("/Font")
        source_fonts = source_fonts.get_object() if source_fonts is not None else {}
        for resource in source_fonts.values():
            assert resource.get_object()["/Subtype"] != "/Type3", "Fresh font42 base required"
        composed = PageObject.create_blank_page(width=width, height=height)

        composed.merge_transformed_page(
            source, Transformation().translate(tx=source_x_offset, ty=source_y_offset),
        )

        overlay_buffer = io.BytesIO()
        pen = canvas.Canvas(overlay_buffer, pagesize=(width, height), initialFontName="FigureCmr10")
        red = (0.80, 0.07, 0.10)
        crop_x, crop_size = 35.0, 224.0
        headers = []
        for role in ("high", "low"):
            spec = bin_specs[role]
            x0, y0, x1, y1 = spec["source_pdf_rect"]
            x0 += source_x_offset
            x1 += source_x_offset
            y0 += source_y_offset
            y1 += source_y_offset
            # Both connectors run below their 0.8/0.2 tick's text bounds.
            # Both connector heights are outside the vertical y-axis title.
            attachment_fraction = 0.3
            center_y = y0 + attachment_fraction * (y1 - y0)
            crop_y = center_y - crop_size / 2

            pen.setStrokeColorRGB(*red)
            pen.setLineWidth(2.7)
            pen.rect(x0, y0, x1 - x0, y1 - y0, fill=0, stroke=1)

            # Connect through the otherwise empty margin, never the data.
            pen.setLineWidth(2.3)
            connector = pen.beginPath()
            connector.moveTo(x0 - 2, center_y)
            connector.lineTo(crop_x + crop_size + 10, center_y)
            pen.drawPath(connector, stroke=1, fill=0)
            arrow = pen.beginPath()
            arrow.moveTo(crop_x + crop_size + 17, center_y + 4)
            arrow.lineTo(crop_x + crop_size + 10, center_y)
            arrow.lineTo(crop_x + crop_size + 17, center_y - 4)
            pen.drawPath(arrow, stroke=1, fill=0)

            # Images themselves have no rectangles or other annotations.
            pen.drawImage(str(crops[role]), crop_x, crop_y, crop_size, crop_size)
            headers.append((
                crop_x + crop_size / 2, crop_y + crop_size + 17,
                f"Object ratio: {spec['actual_ratio']:.4f}",
            ))
        pen.save()
        overlay_buffer.seek(0)
        composed.merge_page(PdfReader(overlay_buffer).pages[0])
        header_buffer = make_header_overlay(width, height, headers)
        header_page = PdfReader(header_buffer).pages[0]
        assert abs(float(header_page.mediabox.width) - width) < 1e-6
        assert abs(float(header_page.mediabox.height) - height) < 1e-6
        composed.merge_page(header_page)
        writer = PdfWriter()
        writer.add_page(composed)
        writer.add_metadata({
            "/Title": "Figure 4 - object ratio bin examples",
            "/Subject": "New original-12 inference heatmap with unchanged exact-bin examples",
        })
        with destination.open("wb") as stream:
            writer.write(stream)
        return {
            "page_size_points": [width, height],
            "original_figure_transform": {
                "scale": 1, "translate": [source_x_offset, source_y_offset],
            },
            "original_figure_clip": [0, 0, 1008, 720],
            "colorbar_moved_separately": False,
            "photos_left_of_object_ratio_axis": True,
            "connector_attachment_within_bin": {"high": 0.3, "low": 0.3},
            "photo_to_axis_gap_reduction_points": 90,
            "bottom_caption_removed": True,
            "bottom_reserved_space_removed_points": 48,
            "header_font": {
                "family": "cmr10", "backend": "Matplotlib PDF fonttype42 (embedded TrueType)",
                "size_points": 22.8, "style": "normal", "weight": "normal",
                "grayscale": 0.15, "matches_original_axis_labels": True,
            },
            "heatmap_cells": 300,
            "fresh_base_heatmap_data_and_color_mapping_changed_by_composition": False,
            "historical_heatmap_numeric_identity_claimed": False,
            "unused_reportlab_helvetica_preamble_avoided": True,
            "crop_size_points": crop_size,
            "image_rectangles": False,
        }

    def render_png(pdf_path, png_path):
        """Render the finished PDF, so the preview matches the deliverable."""
        document = pdfium.PdfDocument(str(pdf_path))
        page = document[0]
        bitmap = page.render(scale=2)
        bitmap.to_pil().save(png_path)
        bitmap.close()
        page.close()
        document.close()

    return plot_figure4, render_png


@app.cell
def _(
    bin_specs, correct_cmsy10_minus_mapping, hashlib, input_hashes, inputs, json, mo, plot_figure4,
    render_png, root,
):
    output_dir = root.parent / "staging/dim/practical/imagenet"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / "spuriosity_orig_modified.pdf"
    output_png = output_dir / "spuriosity_orig_modified.png"
    raw_dir = root.parent / "raw_plot_pdf"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_pdf = raw_dir / output_pdf.name
    composition = plot_figure4(inputs["source"], inputs, bin_specs, raw_pdf)
    unicode_correction = correct_cmsy10_minus_mapping(raw_pdf, output_pdf)
    render_png(output_pdf, output_png)
    assert input_hashes == {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in inputs.items()
    }
    record = {
        "unicode_correction": unicode_correction,
        "original_inputs_unchanged": True,
        "source_hashes": input_hashes,
        "bin_annotations": bin_specs,
        "composition": composition,
        "ratio_definition": "Figure4 bbox geometry ratio; not PIL binary-mask ratio",
        "interpretation": "Illustrative inputs for the bins, not individual model predictions",
        "training_or_gpu_computation": False,
        "exports": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in (output_pdf, output_png)],
        "status": "exported_pending_visual_QA",
    }
    (root / "composition_record.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8",
    )
    mo.image(str(output_png))
    return


if __name__ == "__main__":
    app.run()
