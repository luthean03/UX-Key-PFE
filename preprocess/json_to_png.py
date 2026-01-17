import json
import os
import numpy as np
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
INPUT_FOLDER = "./archetypes"   # Mettez vos json (vieux et nouveaux) ici
OUTPUT_FOLDER = "./arch_png"  # Dossier de sortie
# =================================================

def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def get_node_geometry(node):
    """
    Fonction polymorphe : Récupère la géométrie (x, y, w, h)
    peu importe le format du JSON.
    """
    # CAS 1 : NOUVEAU FORMAT (bounds: {x, y, width, height})
    if 'bounds' in node:
        b = node['bounds']
        # On utilise .get() pour éviter les erreurs si une clé manque
        return (
            int(b.get('x', 0)), 
            int(b.get('y', 0)), 
            int(b.get('width', 0)), 
            int(b.get('height', 0))
        )
    
    # CAS 2 : ANCIEN FORMAT (b: [x, y, w, h])
    elif 'b' in node:
        b = node['b']
        return (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
    
    return None

def get_node_children(node):
    """
    Fonction polymorphe : Récupère la liste des enfants
    """
    if 'children' in node:
        return node['children'] # Nouveau format
    elif 'c' in node:
        return node['c'] # Ancien format
    return []

def paint_additive_recursive(node, canvas, parent_x, parent_y, width, height):
    """
    Logique de peinture additive agnostique du format
    """
    geometry = get_node_geometry(node)
    
    if geometry:
        rel_x, rel_y, w, h = geometry
        
        # Calcul des coordonnées absolues
        abs_x = parent_x + rel_x
        abs_y = parent_y + rel_y
        
        # Clipping (Sécurité pour ne pas dessiner hors du tableau numpy)
        y_start = max(0, int(abs_y))
        y_end = min(height, int(abs_y + h))
        x_start = max(0, int(abs_x))
        x_end = min(width, int(abs_x + w))
        
        # === LOGIQUE ADDITIVE (+1 par couche) ===
        if y_end > y_start and x_end > x_start:
            canvas[y_start:y_end, x_start:x_end] += 1.0 
        
        # Récursion sur les enfants
        children = get_node_children(node)
        for child in children:
            paint_additive_recursive(child, canvas, abs_x, abs_y, width, height)

def process_file(file_path, output_dir):
    filename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # === DETECTION DE LA RACINE ET DES DIMENSIONS ===
        root_node = None
        w_canvas, h_canvas = 0, 0

        # CAS 1 : Ancien Format (racine souvent dans "r", dimensions à la racine)
        if 'r' in data and 'w' in data:
            root_node = data['r']
            w_canvas = int(data['w'])
            h_canvas = int(data['h'])
            
        # CAS 2 : Nouveau Format (racine est l'objet lui-même ou dans "bounds")
        elif 'bounds' in data:
            root_node = data # Le fichier entier est le root node
            w_canvas = int(data['bounds']['width'])
            h_canvas = int(data['bounds']['height'])
            
        else:
            print(f"⚠️  Format inconnu pour : {filename}")
            return

        # Création du canvas
        canvas = np.zeros((h_canvas, w_canvas), dtype=np.float32)
        
        # Lancement de la peinture
        # Note : On commence à (0,0) car les coord du root sont souvent relatives à la page
        paint_additive_recursive(root_node, canvas, 0, 0, w_canvas, h_canvas)
        
        # === POST-TRAITEMENT ET SAUVEGARDE (Identique au script précédent) ===
        max_depth = np.max(canvas)
        if max_depth > 0:
            img_linear = canvas / max_depth
        else:
            img_linear = canvas

        # Gamma encoding pour la visu humaine
        img_contrast = np.power(img_linear, 2.0)
        
        # Sauvegarde
        output_path_linear = os.path.join(output_dir, f"{name_without_ext}_linear.png")
        plt.imsave(output_path_linear, img_linear, cmap='gray', vmin=0.0, vmax=1.0)
        
        output_path_contrast = os.path.join(output_dir, f"{name_without_ext}_visu.png")
        plt.imsave(output_path_contrast, img_contrast, cmap='gray', vmin=0.0, vmax=1.0)
        
        print(f"✅ Traité : {filename} (Format {'Nouveau' if 'bounds' in data else 'Ancien'})")

    except Exception as e:
        print(f"❌ Erreur sur {filename} : {e}")

def main():
    print("🚀 Traitement hybride des JSON...")
    ensure_folder_exists(OUTPUT_FOLDER)
    
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith('.json')]
    
    if not files:
        print(f"Aucun fichier trouvé dans {INPUT_FOLDER}")
        return

    for file in files:
        full_path = os.path.join(INPUT_FOLDER, file)
        process_file(full_path, OUTPUT_FOLDER)

if __name__ == "__main__":
    main()