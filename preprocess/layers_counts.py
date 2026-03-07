import os
import json
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import matplotlib.pyplot as plt

INPUT_FOLDER = "./dataset_supervised/dataset_supervised_pc/json"
MAX_WORKERS = min(8, multiprocessing.cpu_count() * 2)


def get_node_geometry(node):
    if 'bounds' in node:
        b = node['bounds']
        return (
            int(b.get('x', 0)),
            int(b.get('y', 0)),
            int(b.get('width', 0)),
            int(b.get('height', 0))
        )
    elif 'b' in node:
        b = node['b']
        return (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
    return None


def get_node_children(node):
    if 'children' in node:
        return node['children']
    elif 'c' in node:
        return node['c']
    return []


def find_min_y_coord(node, abs_y, current_min_y):
    geometry = get_node_geometry(node)
    node_y = abs_y
    if geometry:
        _, rel_y, _, _ = geometry
        node_y = abs_y + rel_y
        current_min_y = min(current_min_y, node_y)
    for child in get_node_children(node):
        current_min_y = find_min_y_coord(child, node_y, current_min_y)
    return current_min_y


def paint_additive_recursive(node, canvas, parent_x, parent_y, width, height):
    geometry = get_node_geometry(node)
    abs_x, abs_y = parent_x, parent_y
    if geometry:
        rel_x, rel_y, cur_w, cur_h = geometry
        abs_x = parent_x + rel_x
        abs_y = parent_y + rel_y
        x_start = max(0, int(abs_x))
        y_start = max(0, int(abs_y))
        x_end = min(width, int(abs_x + cur_w))
        y_end = min(height, int(abs_y + cur_h))
        if y_end > y_start and x_end > x_start:
            canvas[y_start:y_end, x_start:x_end] += 1.0
    for child in get_node_children(node):
        paint_additive_recursive(child, canvas, abs_x, abs_y, width, height)


def find_content_bounds(heatmap):
    non_zero_rows = np.any(heatmap > 0, axis=1)
    non_zero_cols = np.any(heatmap > 0, axis=0)
    if not np.any(non_zero_rows):
        return None
    row_indices = np.where(non_zero_rows)[0]
    col_indices = np.where(non_zero_cols)[0]
    return (row_indices[0], row_indices[-1] + 1, col_indices[0], col_indices[-1] + 1)


def _render_max_depth(root_node, w_canvas, h_canvas):
    """Render a single LOM/tree and return its max layer depth (int), or 0 on failure."""
    if w_canvas <= 0 or h_canvas <= 0:
        return 0
    min_y = find_min_y_coord(root_node, 0, 0)
    offset_y = abs(min_y) if min_y < 0 else 0
    real_h = h_canvas + offset_y
    heatmap = np.zeros((real_h, w_canvas), dtype=np.float32)
    paint_additive_recursive(root_node, heatmap, 0, offset_y, w_canvas, real_h)
    if np.max(heatmap) == 0:
        return 0
    bounds = find_content_bounds(heatmap)
    if bounds is None:
        return 0
    top_row, bottom_row, _, _ = bounds
    cropped_heatmap = heatmap[top_row:bottom_row, :]
    return int(np.max(cropped_heatmap))


def get_max_depth(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # --- Format 1 : session JSON (json_to_png_supervised_session.py) ---
        # { "isMobile": bool, "loms": { hash: { "root": ..., "pageWidth": ..., "pageHeight": ... } } }
        if 'loms' in data:
            depths = []
            for lom_data in data['loms'].values():
                root_node = lom_data.get('root')
                w_canvas = int(lom_data.get('pageWidth', 0))
                h_canvas = int(lom_data.get('pageHeight', 0))
                if root_node is None:
                    continue
                d = _render_max_depth(root_node, w_canvas, h_canvas)
                if d > 0:
                    depths.append(d)
            return max(depths) if depths else 0

        # --- Format 2 : compact VAE JSON (json_to_png.py) ---
        # { "r": root_node, "w": width, "h": height }
        elif 'r' in data and 'w' in data:
            return _render_max_depth(data['r'], int(data['w']), int(data['h']))

        # --- Format 3 : verbose single-node JSON ---
        # { "bounds": { "x", "y", "width", "height" }, "children": [...] }
        elif 'bounds' in data:
            return _render_max_depth(data, int(data['bounds']['width']), int(data['bounds']['height']))

        return 0
    except Exception:
        return 0


def analyze_dataset():
    files = [os.path.join(INPUT_FOLDER, f) for f in os.listdir(INPUT_FOLDER) if f.endswith('.json')]
    print(f"Fichiers JSON trouvés : {len(files)}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(get_max_depth, files), total=len(files), desc="Analyse des profondeurs"))

    max_depths = [r for r in results if r > 0]

    if not max_depths:
        print("Aucune donnée valide trouvée.")
        return

    p90  = np.percentile(max_depths, 90)
    p95  = np.percentile(max_depths, 95)
    p99  = np.percentile(max_depths, 99)
    abs_max = np.max(max_depths)

    print("\n=== ANALYSE DE LA PROFONDEUR DES WIREFRAMES ===")
    print(f"Fichiers valides       : {len(max_depths)} / {len(files)}")
    print(f"Profondeur moyenne     : {np.mean(max_depths):.2f} couches")
    print(f"Médiane (50%)          : {np.percentile(max_depths, 50):.0f} couches")
    print(f"90ème percentile       : {p90:.0f} couches")
    print(f"95ème percentile       : {p95:.0f} couches")
    print(f"99ème percentile (rec) : {p99:.0f} couches  <-- Valeur recommandée pour GLOBAL_MAX_LAYERS")
    print(f"Maximum absolu         : {abs_max} couches")
    print(f"\n>>> Recommandation : fixer GLOBAL_MAX_LAYERS = {int(p99)}")

    # Histogramme de la distribution
    plt.figure(figsize=(10, 5))
    plt.hist(max_depths, bins=50, color='steelblue', edgecolor='black')
    plt.axvline(p99, color='red', linestyle='--', linewidth=2, label=f'99e percentile = {int(p99)}')
    plt.axvline(p95, color='orange', linestyle='--', linewidth=1.5, label=f'95e percentile = {int(p95)}')
    plt.xlabel("Profondeur maximale (nombre de couches)")
    plt.ylabel("Nombre de fichiers")
    plt.title("Distribution de la profondeur maximale des wireframes")
    plt.legend()
    plt.tight_layout()
    hist_path = os.path.join(os.path.dirname(INPUT_FOLDER), "depth_distribution.png")
    plt.savefig(hist_path)
    print(f"\nHistogramme sauvegardé : {hist_path}")


if __name__ == "__main__":
    analyze_dataset()