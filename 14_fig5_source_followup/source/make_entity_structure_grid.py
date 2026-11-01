import csv
import math
import os
import re

from PIL import Image, ImageDraw, ImageFont


PATTERN = re.compile(r"^(?P<entity>\d+)unique_structure_(?P<structure>\d+)\.bmp$", re.I)
CSV_PATH = "unique_structures_result.csv"
TRANSPOSE = False
MAX_STRUCTURES_PER_ENTITY = 3
COL_PADDING = 24
ROW_PADDING = 24
GRID_LINE_WIDTH = 10
GRID_LINE_COLOR = (0, 0, 0)
HIGHLIGHT_COLOR = (255, 140, 0)
HIGHLIGHT_PROB_THRESHOLD = 0.40


def find_images():
    files = {}
    entities = set()
    structures = set()
    for name in os.listdir("."):
        match = PATTERN.match(name)
        if not match:
            continue
        entity = int(match.group("entity"))
        structure = int(match.group("structure"))
        files[(entity, structure)] = name
        entities.add(entity)
        structures.add(structure)
    return files, sorted(entities), structures


def load_counts(csv_path):
    if not os.path.exists(csv_path):
        return {}, set(), set()

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        entity_cols = [c for c in headers if c.startswith("entity_")]
        entities = set()
        counts = {}
        structures = set()

        for col in entity_cols:
            parts = col.split("_", 1)
            if len(parts) != 2:
                continue
            try:
                entities.add(int(parts[1]))
            except ValueError:
                continue

        for row in reader:
            for col in entity_cols:
                val = row.get(col, "")
                if val is None or val == "":
                    continue
                try:
                    num = float(val)
                except ValueError:
                    continue
                structure = int(round(num))
                try:
                    entity = int(col.split("_", 1)[1])
                except ValueError:
                    continue
                counts[(entity, structure)] = counts.get((entity, structure), 0) + 1
                structures.add(structure)

    return counts, entities, structures


def compute_layout(rows, cols, sample_size):
    sample_w, sample_h = sample_size
    col_padding = COL_PADDING
    row_padding = ROW_PADDING
    left_margin = 70
    top_margin = 50
    right_margin = 20
    bottom_margin = max(20, row_padding)

    max_total_width = 4800
    available = max_total_width - left_margin - right_margin - col_padding * (cols - 1)
    tile_w = max(60, min(sample_w, available // cols))
    tile_h = max(40, int(tile_w * sample_h / sample_w))

    grid_w = left_margin + cols * tile_w + (cols - 1) * col_padding + right_margin
    grid_h = top_margin + rows * tile_h + (rows - 1) * row_padding + bottom_margin

    return {
        "cols": cols,
        "rows": rows,
        "col_padding": col_padding,
        "row_padding": row_padding,
        "left_margin": left_margin,
        "top_margin": top_margin,
        "right_margin": right_margin,
        "bottom_margin": bottom_margin,
        "tile_w": tile_w,
        "tile_h": tile_h,
        "grid_w": grid_w,
        "grid_h": grid_h,
    }


def load_arial(size):
    for path in ("C:\\Windows\\Fonts\\arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def select_top_structures(counts, entity, candidates, max_items):
    scored = []
    for structure in candidates:
        count = counts.get((entity, structure), 0)
        if count > 0:
            scored.append((structure, count))

    if scored:
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [structure for structure, _ in scored[:max_items]]

    return list(candidates)[:max_items]


def order_structures_centered(structures):
    n = len(structures)
    if n <= 1:
        return list(structures)

    offsets = [0, -1, 1, -2, 2]
    center = n // 2
    ordered = [None] * n
    for i, structure in enumerate(structures):
        pos = center + offsets[i]
        if 0 <= pos < n:
            ordered[pos] = structure
    return [structure for structure in ordered if structure is not None]


def build_row_columns(entities, entity_structures, counts, max_items):
    row_cols = {}
    for entity in entities:
        candidates = sorted(entity_structures.get(entity, []))
        selected = select_top_structures(counts, entity, candidates, max_items)
        row_cols[entity] = order_structures_centered(selected)
    return row_cols


def compute_row_lefts(layout, row_items, row_cols):
    total_w = layout["cols"] * layout["tile_w"] + (layout["cols"] - 1) * layout["col_padding"]
    row_lefts = {}
    for row_item in row_items:
        cols = row_cols.get(row_item, [])
        row_w = len(cols) * layout["tile_w"] + max(0, len(cols) - 1) * layout["col_padding"]
        offset = (total_w - row_w) // 2 if total_w > row_w else 0
        row_lefts[row_item] = layout["left_margin"] + offset
    return row_lefts


def draw_labels(draw, row_items, col_items, layout, font, row_tops=None, show_col_labels=True):
    if show_col_labels:
        for col, col_label in enumerate(col_items):
            x = layout["left_margin"] + col * (layout["tile_w"] + layout["col_padding"])
            x += layout["tile_w"] // 2
            draw.text((x, layout["top_margin"] - 20), str(col_label), anchor="mm", font=font, fill="black")

    for row, row_label in enumerate(row_items):
        if row_tops is None:
            y = layout["top_margin"] + row * (layout["tile_h"] + layout["row_padding"])
        else:
            y = row_tops[row]
        y += layout["tile_h"] // 2
        draw.text((layout["left_margin"] - 10, y), str(row_label), anchor="rm", font=font, fill="black")


def paste_images(canvas, files, row_items, row_cols, row_lefts, cell_key, layout, cache=None, row_tops=None):
    if cache is None:
        cache = {}
    for row, row_item in enumerate(row_items):
        row_left = row_lefts[row_item]
        cols = row_cols.get(row_item, [])
        for col, col_item in enumerate(cols):
            key = cell_key(row_item, col_item)
            if key not in files:
                continue
            img = cache.get(key)
            if img is None:
                img = Image.open(files[key])
                if img.size != (layout["tile_w"], layout["tile_h"]):
                    img = img.resize((layout["tile_w"], layout["tile_h"]), Image.LANCZOS)
                cache[key] = img
            x = row_left + col * (layout["tile_w"] + layout["col_padding"])
            if row_tops is None:
                y = layout["top_margin"] + row * (layout["tile_h"] + layout["row_padding"])
            else:
                y = row_tops[row]
            canvas.paste(img, (x, y))
    return cache


def draw_grid(
    draw,
    row_items,
    row_cols,
    row_lefts,
    layout,
    cell_key=None,
    highlight_cells=None,
    highlight_color=HIGHLIGHT_COLOR,
    line_color=GRID_LINE_COLOR,
    row_tops=None,
    cell_h=None,
):
    cell_h = layout["tile_h"] if cell_h is None else cell_h
    for row, row_item in enumerate(row_items):
        row_left = row_lefts[row_item]
        cols = row_cols.get(row_item, [])
        for col, col_item in enumerate(cols):
            x = row_left + col * (layout["tile_w"] + layout["col_padding"])
            if row_tops is None:
                y = layout["top_margin"] + row * (layout["tile_h"] + layout["row_padding"])
            else:
                y = row_tops[row]
            outline = line_color
            if highlight_cells:
                key = cell_key(row_item, col_item) if cell_key else (row_item, col_item)
                if key in highlight_cells:
                    outline = highlight_color
            draw.rectangle(
                [x, y, x + layout["tile_w"] - 1, y + cell_h - 1],
                outline=outline,
                width=GRID_LINE_WIDTH,
            )


def apply_overlay(canvas, row_items, row_cols, row_lefts, cell_key, layout, counts, max_count):
    if max_count <= 0:
        return canvas
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    log_max = math.log1p(max_count)
    base_color = (0, 120, 215)

    for row, row_item in enumerate(row_items):
        row_left = row_lefts[row_item]
        cols = row_cols.get(row_item, [])
        for col, col_item in enumerate(cols):
            count = counts.get(cell_key(row_item, col_item), 0)
            if count <= 0:
                continue
            norm = math.log1p(count) / log_max if log_max > 0 else 0.0
            alpha = int(40 + 180 * norm)
            x = row_left + col * (layout["tile_w"] + layout["col_padding"])
            y = layout["top_margin"] + row * (layout["tile_h"] + layout["row_padding"])
            draw.rectangle(
                [x, y, x + layout["tile_w"] - 1, y + layout["tile_h"] - 1],
                fill=(base_color[0], base_color[1], base_color[2], alpha),
            )

    combined = Image.alpha_composite(canvas.convert("RGBA"), overlay)
    return combined.convert("RGB")


def compute_bar_layout(layout, rows):
    bar_h = max(8, layout["tile_h"] // 8)
    bar_gap = 4
    row_gap = max(8, layout["row_padding"])
    row_tops = []
    bar_tops = []

    y = layout["top_margin"]
    for _ in range(rows):
        row_tops.append(y)
        bar_tops.append(y + layout["tile_h"] + bar_gap)
        y += layout["tile_h"] + bar_gap + bar_h + row_gap

    grid_h = y - row_gap + layout["bottom_margin"]
    return bar_h, row_gap, row_tops, bar_tops, grid_h


def draw_distribution_bars(draw, row_items, row_cols, row_lefts, cell_key, layout, counts, max_count, bar_tops, bar_h):
    if max_count <= 0:
        return
    log_max = math.log1p(max_count)
    base_color = (0, 120, 215)

    for row, row_item in enumerate(row_items):
        y = bar_tops[row]
        row_left = row_lefts[row_item]
        cols = row_cols.get(row_item, [])
        for col, col_item in enumerate(cols):
            count = counts.get(cell_key(row_item, col_item), 0)
            if count <= 0:
                continue
            norm = math.log1p(count) / log_max if log_max > 0 else 0.0
            intensity = int(50 + 180 * norm)
            color = (base_color[0], base_color[1], base_color[2], intensity)
            x = row_left + col * (layout["tile_w"] + layout["col_padding"])
            draw.rectangle(
                [x, y, x + layout["tile_w"] - 1, y + bar_h - 1],
                fill=color,
            )


def draw_probabilities(
    draw,
    canvas,
    row_items,
    row_cols,
    row_lefts,
    cell_key,
    layout,
    counts,
    row_totals,
    row_tops=None,
    font=None,
    placement="top_left",
    background_alpha=160,
    background_padding=2,
):
    if font is None:
        font = ImageFont.load_default()
    for row, row_item in enumerate(row_items):
        total = row_totals.get(row_item, 0)
        if total <= 0:
            continue
        row_left = row_lefts[row_item]
        cols = row_cols.get(row_item, [])
        if row_tops is None:
            row_top = layout["top_margin"] + row * (layout["tile_h"] + layout["row_padding"])
        else:
            row_top = row_tops[row]
        for col, col_item in enumerate(cols):
            count = counts.get(cell_key(row_item, col_item), 0)
            if count <= 0:
                continue
            prob = count / total
            text = f"{prob:.1%}"
            x = row_left + col * (layout["tile_w"] + layout["col_padding"]) + 2
            if placement == "below":
                y = row_top + layout["tile_h"] + 2
            else:
                y = row_top + 2
            if canvas is not None and background_alpha > 0:
                bbox = draw.textbbox((x, y), text, font=font, stroke_width=1)
                bg_left = bbox[0] - background_padding
                bg_top = bbox[1] - background_padding
                bg_right = bbox[2] + background_padding
                bg_bottom = bbox[3] + background_padding
                bg_w = max(1, int(bg_right - bg_left))
                bg_h = max(1, int(bg_bottom - bg_top))
                bg = Image.new("RGBA", (bg_w, bg_h), (255, 255, 255, background_alpha))
                canvas.paste(bg, (int(bg_left), int(bg_top)), bg)
            draw.text(
                (x, y),
                text,
                font=font,
                fill="black",
                stroke_width=1,
                stroke_fill="white",
            )


def main(transpose=TRANSPOSE):
    files, image_entities, image_structures = find_images()
    if not files:
        raise SystemExit("No matching BMP files found.")

    counts, csv_entities, csv_structures = load_counts(CSV_PATH)
    entities = image_entities
    structures = set(image_structures)
    structures.update(csv_structures)
    if not structures:
        raise SystemExit("No structures found.")

    structure_range = list(range(1, max(structures) + 1))

    if transpose:
        raise SystemExit("This layout requires entities as rows; set TRANSPOSE = False.")

    row_items = entities
    col_items = structure_range

    def cell_key(row_item, col_item):
        return (row_item, col_item)

    entity_structures = {}
    for entity, structure in files:
        entity_structures.setdefault(entity, set()).add(structure)

    sample = Image.open(next(iter(files.values())))
    row_cols = build_row_columns(row_items, entity_structures, counts, MAX_STRUCTURES_PER_ENTITY)
    max_cols = max((len(cols) for cols in row_cols.values()), default=0)
    if max_cols <= 0:
        raise SystemExit("No structures selected for display.")
    layout = compute_layout(len(row_items), max_cols, sample.size)
    row_lefts = compute_row_lefts(layout, row_items, row_cols)
    font = ImageFont.load_default()
    prob_font = load_arial(max(9, layout["tile_h"] // 7))

    max_count = max(counts.values(), default=0)
    row_totals = {}
    for (entity, _structure), count in counts.items():
        row_totals[entity] = row_totals.get(entity, 0) + count
    highlight_cells = set()
    for entity in row_items:
        if entity <= 5:
            continue
        total = row_totals.get(entity, 0)
        if total <= 0:
            continue
        cols = row_cols.get(entity, [])
        if not cols:
            continue
        for structure in cols:
            count = counts.get((entity, structure), 0)
            if total > 0 and (count / total) >= HIGHLIGHT_PROB_THRESHOLD:
                highlight_cells.add(cell_key(entity, structure))

    cache = {}

    # Base grid with lines
    base = Image.new("RGB", (layout["grid_w"], layout["grid_h"]), "white")
    draw = ImageDraw.Draw(base)
    draw_labels(draw, row_items, col_items, layout, font, show_col_labels=False)
    cache = paste_images(base, files, row_items, row_cols, row_lefts, cell_key, layout, cache)
    draw_probabilities(
        draw,
        base,
        row_items,
        row_cols,
        row_lefts,
        cell_key,
        layout,
        counts,
        row_totals,
        font=prob_font,
    )
    draw_grid(
        draw,
        row_items,
        row_cols,
        row_lefts,
        layout,
        cell_key=cell_key,
        highlight_cells=highlight_cells,
    )
    base.save("entity_structure_grid.png")

    # A: overlay heatmap on top of images
    overlay = Image.new("RGB", (layout["grid_w"], layout["grid_h"]), "white")
    draw = ImageDraw.Draw(overlay)
    draw_labels(draw, row_items, col_items, layout, font, show_col_labels=False)
    paste_images(overlay, files, row_items, row_cols, row_lefts, cell_key, layout, cache)
    overlay = apply_overlay(overlay, row_items, row_cols, row_lefts, cell_key, layout, counts, max_count)
    draw = ImageDraw.Draw(overlay)
    draw_probabilities(draw, overlay, row_items, row_cols, row_lefts, cell_key, layout, counts, row_totals)
    draw_grid(
        draw,
        row_items,
        row_cols,
        row_lefts,
        layout,
        cell_key=cell_key,
        highlight_cells=highlight_cells,
    )
    overlay.save("entity_structure_grid_overlay.png")

    # B: distribution bars under each row
    bar_h, row_gap, row_tops, bar_tops, grid_h = compute_bar_layout(layout, len(row_items))
    bar_canvas = Image.new("RGB", (layout["grid_w"], grid_h), "white")
    bar_draw = ImageDraw.Draw(bar_canvas, "RGBA")
    draw_labels(bar_draw, row_items, col_items, layout, font, row_tops=row_tops, show_col_labels=False)
    paste_images(bar_canvas, files, row_items, row_cols, row_lefts, cell_key, layout, cache, row_tops=row_tops)
    draw_grid(bar_draw, row_items, row_cols, row_lefts, layout, row_tops=row_tops)
    draw_distribution_bars(
        bar_draw, row_items, row_cols, row_lefts, cell_key, layout, counts, max_count, bar_tops, bar_h
    )
    draw_grid(bar_draw, row_items, row_cols, row_lefts, layout, row_tops=bar_tops, cell_h=bar_h)
    draw_probabilities(
        bar_draw,
        bar_canvas,
        row_items,
        row_cols,
        row_lefts,
        cell_key,
        layout,
        counts,
        row_totals,
        row_tops=row_tops,
    )
    bar_canvas.save("entity_structure_grid_bars.png")

    print("Saved entity_structure_grid.png")
    print("Saved entity_structure_grid_overlay.png")
    print("Saved entity_structure_grid_bars.png")


if __name__ == "__main__":
    main()
