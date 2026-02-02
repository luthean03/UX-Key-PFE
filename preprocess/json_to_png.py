import json
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Configuration
INPUT_FOLDER = "./dataset/archetypes_phone/json"

OUTPUT_FOLDER = "dataset/archetypes_phone/png"
MAX_WORKERS = min(8, multiprocessing.cpu_count() * 2)

def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

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
    cur_w, cur_h = 0, 0
    
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


def process_file(file_path, output_dir):
    """
    Process a single JSON file.
    """
    filename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        root_node = None
        w_canvas, h_canvas = 0, 0

        if 'r' in data and 'w' in data:
            root_node = data['r']
            w_canvas = int(data['w'])
            h_canvas = int(data['h'])
        elif 'bounds' in data:
            root_node = data
            w_canvas = int(data['bounds']['width'])
            h_canvas = int(data['bounds']['height'])
        else:
            return False, f"Unknown format: {filename}"

        min_y = find_min_y_coord(root_node, 0, 0)
        offset_x = 0
        offset_y = abs(min_y) if min_y < 0 else 0

        real_h = h_canvas + offset_y
        heatmap = np.zeros((real_h, w_canvas), dtype=np.float32)
        
        paint_additive_recursive(root_node, heatmap, offset_x, offset_y, w_canvas, real_h)

        if np.max(heatmap) == 0:
            return False, f"Empty: {filename}"

        bounds = find_content_bounds(heatmap)
        if bounds is None:
            return False, f"No content found: {filename}"
        
        top_row, bottom_row, left_col, right_col = bounds
        cropped_heatmap = heatmap[top_row:bottom_row, :]
        rows_cropped = heatmap.shape[0] - cropped_heatmap.shape[0]

        local_max = np.max(cropped_heatmap)
        if local_max == 0:
            return False, "Max value is 0"

        normalized_img = cropped_heatmap / local_max
        final_img_uint8 = (normalized_img * 255.0).astype(np.uint8)

        # === NEW FILENAME LOGIC ===
        # Check if '-None' is already present (case-insensitive)
        name_lower = name_without_ext.lower()
        has_none_suffix = '-none' in name_lower
        
        # Check if filename contains NO letters (only digits and special chars like -, _)
        has_no_letters = not any(c.isalpha() for c in name_without_ext)
        
        # Add '-None' only if: does NOT have '-None' AND has NO letters
        if not has_none_suffix and has_no_letters:
            suffix = "-None"
        else:
            suffix = ""
        
        final_filename = f"{name_without_ext}{suffix}.png"
        output_path = os.path.join(output_dir, final_filename)
        # ==========================

        Image.fromarray(final_img_uint8, mode='L').save(output_path)
        
        return True, rows_cropped

    except Exception as e:
        return False, f"Error {filename}: {e}"



def main():
    ensure_folder_exists(OUTPUT_FOLDER)
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.json')]
    
    print(f"Found {len(files)} JSON files...")
    print(f"Using {MAX_WORKERS} workers")
    
    successful = 0
    failed = 0
    total_rows_cropped = 0
    errors = []
    
    file_paths = [(os.path.join(INPUT_FOLDER, f), OUTPUT_FOLDER) for f in files]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {
            executor.submit(process_file, *fp): os.path.basename(fp[0])
            for fp in file_paths
        }
        
        with tqdm(total=len(files), unit="img") as pbar:
            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    success, result = future.result()
                    if success:
                        successful += 1
                        total_rows_cropped += result
                    else:
                        failed += 1
                        errors.append(f"{filename}: {result}")
                except Exception as e:
                    failed += 1
                    errors.append(f"{filename}: {e}")
                
                pbar.update(1)
    
    print("\n" + "=" * 60)
    print(f"Done. Success: {successful}, Failed: {failed}")
    if errors:
        print("First 5 Errors:")
        for e in errors[:5]: print(e)

if __name__ == "__main__":
    main()