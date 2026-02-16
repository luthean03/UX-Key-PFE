import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def scale_image(image_path, output_dir, scale_factor):
    try:
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
        
        with Image.open(image_path) as img:
            new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
            img_resized = img.resize(new_size, resample=Image.LANCZOS)
            # Save to new output directory
            img_resized.save(output_path, format="PNG")
    except Exception:
        pass  # (optional: log errors)

def batch_scale_images(image_dir, output_dir, scale_factor=0.5, max_workers=12):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Only PNG images in the directory
    image_files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, f)) and f.lower().endswith('.png')
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for image_path in image_files:
            futures.append(executor.submit(scale_image, image_path, output_dir, scale_factor))

        for _ in tqdm(as_completed(futures), total=len(futures), desc="Scaling Images"):
            pass

if __name__ == "__main__":
    image_dir = r"/usr/users/sdim/sdim_31/UX-Key-PFE/dataset/archetypes_phone/png"
    output_dir = r"/usr/users/sdim/sdim_31/UX-Key-PFE//dataset/archetypes_phone/png_scaled"
    batch_scale_images(image_dir, output_dir, scale_factor=0.2, max_workers=8)
