import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Configuration
INPUT_FOLDER = "./archetypes"   # Put your JSON files (old and new formats) here
OUTPUT_FOLDER = "./arch_png"    # Output directory

def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def get_node_geometry(node):
    """
    Polymorphic function: Extract geometry (x, y, w, h)
    regardless of JSON format.
    """
    # CASE 1: NEW FORMAT (bounds: {x, y, width, height})
    if 'bounds' in node:
        b = node['bounds']
        # Use .get() to avoid errors if key is missing
        return (
            int(b.get('x', 0)), 
            int(b.get('y', 0)), 
            int(b.get('width', 0)), 
            int(b.get('height', 0))
        )
    
    # CASE 2: OLD FORMAT (b: [x, y, w, h])
    elif 'b' in node:
        b = node['b']
        return (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
    
    return None

def get_node_children(node):
    """
    Polymorphic function: Extract children list
    """
    if 'children' in node:
        return node['children']  # New format
    elif 'c' in node:
        return node['c']  # Old format
    return []

def paint_additive_recursive(node, canvas, parent_x, parent_y, width, height):
    """
    Format-agnostic additive painting logic
    """
    geometry = get_node_geometry(node)
    
    if geometry:
        rel_x, rel_y, w, h = geometry
        
        # Calculate absolute coordinates
        abs_x = parent_x + rel_x
        abs_y = parent_y + rel_y
        
        # Clip to canvas bounds to prevent out-of-bounds drawing
        y_start = max(0, int(abs_y))
        y_end = min(height, int(abs_y + h))
        x_start = max(0, int(abs_x))
        x_end = min(width, int(abs_x + w))
        
        # Additive logic: increment depth by 1 per layer
        if y_end > y_start and x_end > x_start:
            canvas[y_start:y_end, x_start:x_end] += 1.0 
        
        # Recursively process children
        children = get_node_children(node)
        for child in children:
            paint_additive_recursive(child, canvas, abs_x, abs_y, width, height)

def process_file(file_path, output_dir):
    filename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Detect root node and canvas dimensions
        root_node = None
        w_canvas, h_canvas = 0, 0

        # Case 1: Old format (root in 'r', dimensions at root level)
        if 'r' in data and 'w' in data:
            root_node = data['r']
            w_canvas = int(data['w'])
            h_canvas = int(data['h'])
            
        # Case 2: New format (root is object itself or in 'bounds')
        elif 'bounds' in data:
            root_node = data  # The entire file is the root node
            w_canvas = int(data['bounds']['width'])
            h_canvas = int(data['bounds']['height'])
            
        else:
            print(f"[!] Format inconnu pour : {filename}")
            return

        # Create canvas
        canvas = np.zeros((h_canvas, w_canvas), dtype=np.float32)
        
        # Render wireframe starting at (0,0) as root coords are page-relative
        paint_additive_recursive(root_node, canvas, 0, 0, w_canvas, h_canvas)
        
        # Post-processing and output
        max_depth = np.max(canvas)
        if max_depth > 0:
            img_linear = canvas / max_depth
        else:
            img_linear = canvas

        # Apply gamma encoding for visualization
        img_contrast = np.power(img_linear, 2.0)
        
        # Save outputs
        output_path_linear = os.path.join(output_dir, f"{name_without_ext}_linear.png")
        plt.imsave(output_path_linear, img_linear, cmap='gray', vmin=0.0, vmax=1.0)
        
        output_path_contrast = os.path.join(output_dir, f"{name_without_ext}_visu.png")
        plt.imsave(output_path_contrast, img_contrast, cmap='gray', vmin=0.0, vmax=1.0)
        
        print(f"[OK] Traité : {filename} (Format {'Nouveau' if 'bounds' in data else 'Ancien'})")

    except Exception as e:
        print(f"[X] Erreur sur {filename} : {e}")

def main():
    print("[*] Traitement hybride des JSON...")
    ensure_folder_exists(OUTPUT_FOLDER)
    
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith('.json')]
    
    if not files:
        print(f"No files found in {INPUT_FOLDER}")
        return

    for file in files:
        full_path = os.path.join(INPUT_FOLDER, file)
        process_file(full_path, OUTPUT_FOLDER)

if __name__ == "__main__":
    main()