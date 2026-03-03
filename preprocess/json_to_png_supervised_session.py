"""Convert session JSON files to PNG heatmap images.

Each session JSON contains multiple LOMs (layout object models).
Each LOM produces one PNG. Sessions are classified as phone or pc
based on the isMobile flag, and both input JSONs and output PNGs
are sorted into phone/ and pc/ subdirectories accordingly.
"""

import json
import os
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

INPUT_FOLDER = "./eatennaturallyfell/sessions/daily_zip/2026/2026-02-16"

# Output folders for PNGs
PNG_PC_FOLDER = "./dataset_supervised/dataset_supervised_pc/png"
PNG_PHONE_FOLDER = "./dataset_supervised/dataset_supervised_phone/png"

# Output folders for sorted JSONs
JSON_PC_FOLDER = "./dataset_supervised/dataset_supervised_pc/json"
JSON_PHONE_FOLDER = "./dataset_supervised/dataset_supervised_phone/json"

MAX_WORKERS = min(8, multiprocessing.cpu_count() * 2)


def ensure_folder_exists(folder_path):
    os.makedirs(folder_path, exist_ok=True)


def get_node_geometry(node):
    """
    Polymorphic function: Extract geometry (x, y, w, h).
    """
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
    """
    Finds the minimum Y coordinate to handle negative positioning.
    """
    geometry = get_node_geometry(node)

    node_y = abs_y
    if geometry:
        _, rel_y, _, _ = geometry
        node_y = abs_y + rel_y
        current_min_y = min(current_min_y, node_y)

    children = get_node_children(node)
    for child in children:
        current_min_y = find_min_y_coord(child, node_y, current_min_y)

    return current_min_y


def paint_additive_recursive(node, canvas, parent_x, parent_y, width, height):
    """
    Paints filled rectangles.
    Nesting increases brightness (Additive).
    Strictly clips to canvas width/height.
    """
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

    children = get_node_children(node)
    for child in children:
        paint_additive_recursive(child, canvas, abs_x, abs_y, width, height)


def find_content_bounds(heatmap):
    """
    Find the actual content bounds to crop empty headers/footers.
    """
    non_zero_rows = np.any(heatmap > 0, axis=1)
    non_zero_cols = np.any(heatmap > 0, axis=0)

    if not np.any(non_zero_rows):
        return None

    row_indices = np.where(non_zero_rows)[0]
    col_indices = np.where(non_zero_cols)[0]

    top_row = row_indices[0]
    bottom_row = row_indices[-1] + 1
    left_col = col_indices[0]
    right_col = col_indices[-1] + 1

    return (top_row, bottom_row, left_col, right_col)


def render_lom_to_png(root_node, w_canvas, h_canvas, output_path):
    """
    Render a single LOM tree to a PNG heatmap image.
    Returns (success: bool, info: str).
    """
    min_y = find_min_y_coord(root_node, 0, 0)
    offset_x = 0
    offset_y = abs(min_y) if min_y < 0 else 0

    real_h = h_canvas + offset_y
    heatmap = np.zeros((real_h, w_canvas), dtype=np.float32)

    paint_additive_recursive(root_node, heatmap, offset_x, offset_y, w_canvas, real_h)

    if np.max(heatmap) == 0:
        return False, "Empty heatmap"

    bounds = find_content_bounds(heatmap)
    if bounds is None:
        return False, "No content found"

    top_row, bottom_row, left_col, right_col = bounds
    cropped_heatmap = heatmap[top_row:bottom_row, :]
    rows_cropped = heatmap.shape[0] - cropped_heatmap.shape[0]

    local_max = np.max(cropped_heatmap)
    if local_max == 0:
        return False, "Max value is 0"

    normalized_img = cropped_heatmap / local_max
    final_img_uint8 = (normalized_img * 255.0).astype(np.uint8)

    Image.fromarray(final_img_uint8, mode='L').save(output_path)
    return True, rows_cropped


def process_session(file_path, png_pc_dir, png_phone_dir, json_pc_dir, json_phone_dir):
    """
    Process a single session JSON file.
    Extracts all LOMs, renders each to a PNG, and sorts outputs
    into the appropriate pc or phone directories based on isMobile.
    Also copies the input JSON into the corresponding JSON folder.

    Returns (num_success, num_failed, errors_list).
    """
    filename = os.path.basename(file_path)
    session_id = os.path.splitext(filename)[0]

    num_success = 0
    num_failed = 0
    errors = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        is_mobile = data.get('isMobile', False)
        png_dir = png_phone_dir if is_mobile else png_pc_dir
        json_dir = json_phone_dir if is_mobile else json_pc_dir

        # Copy the input JSON into the sorted folder
        shutil.copy2(file_path, os.path.join(json_dir, filename))

        # Extract LOMs
        loms = data.get('loms', {})
        if not loms:
            return 0, 1, [f"{filename}: No LOMs found"]

        for lom_idx, (lom_hash, lom_data) in enumerate(loms.items()):
            try:
                root_node = lom_data.get('root')
                w_canvas = int(lom_data.get('pageWidth', 0))
                h_canvas = int(lom_data.get('pageHeight', 0))

                if root_node is None or w_canvas == 0 or h_canvas == 0:
                    num_failed += 1
                    errors.append(f"{session_id}/lom_{lom_idx}: Missing root or dimensions")
                    continue

                # Use session_id + lom index as filename
                png_filename = f"{session_id}_{lom_idx}.png"
                output_path = os.path.join(png_dir, png_filename)

                success, info = render_lom_to_png(root_node, w_canvas, h_canvas, output_path)
                if success:
                    num_success += 1
                else:
                    num_failed += 1
                    errors.append(f"{session_id}/lom_{lom_idx}: {info}")

            except Exception as e:
                num_failed += 1
                errors.append(f"{session_id}/lom_{lom_idx}: {e}")

    except Exception as e:
        return 0, 1, [f"{filename}: {e}"]

    return num_success, num_failed, errors


def main():
    # Create all output directories
    for d in (PNG_PC_FOLDER, PNG_PHONE_FOLDER, JSON_PC_FOLDER, JSON_PHONE_FOLDER):
        ensure_folder_exists(d)

    files = [
        f for f in os.listdir(INPUT_FOLDER)
        if f.endswith('.json') and f != 'manifest.json'
    ]

    print(f"Found {len(files)} session JSON files...")
    print(f"Using {MAX_WORKERS} workers")
    print(f"PNG  -> pc: {PNG_PC_FOLDER}  |  phone: {PNG_PHONE_FOLDER}")
    print(f"JSON -> pc: {JSON_PC_FOLDER}  |  phone: {JSON_PHONE_FOLDER}")

    total_success = 0
    total_failed = 0
    all_errors = []

    file_paths = [
        (os.path.join(INPUT_FOLDER, f), PNG_PC_FOLDER, PNG_PHONE_FOLDER, JSON_PC_FOLDER, JSON_PHONE_FOLDER)
        for f in files
    ]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {
            executor.submit(process_session, *fp): os.path.basename(fp[0])
            for fp in file_paths
        }

        with tqdm(total=len(files), unit="session") as pbar:
            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    n_ok, n_fail, errs = future.result()
                    total_success += n_ok
                    total_failed += n_fail
                    all_errors.extend(errs)
                except Exception as e:
                    total_failed += 1
                    all_errors.append(f"{filename}: {e}")

                pbar.update(1)

    print("\n" + "=" * 60)
    print(f"Done. PNG success: {total_success}, PNG failed: {total_failed}")
    print(f"Sessions: {len(files)}")

    # Count output files per category
    for label, png_d, json_d in [("pc", PNG_PC_FOLDER, JSON_PC_FOLDER), ("phone", PNG_PHONE_FOLDER, JSON_PHONE_FOLDER)]:
        n_png = len([f for f in os.listdir(png_d) if f.endswith('.png')]) if os.path.isdir(png_d) else 0
        n_json = len([f for f in os.listdir(json_d) if f.endswith('.json')]) if os.path.isdir(json_d) else 0
        print(f"  {label}: {n_png} PNGs, {n_json} JSONs")

    if all_errors:
        print(f"\nFirst 10 errors (out of {len(all_errors)}):")
        for e in all_errors[:10]:
            print(f"  {e}")


if __name__ == "__main__":
    main()