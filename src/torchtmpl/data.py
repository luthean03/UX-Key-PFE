# src/torchtmpl/data.py
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import os
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random
import logging
import numpy as np
from typing import List, Optional

Image.MAX_IMAGE_PIXELS = None 


class SmartBatchSampler(Sampler):
    """Batches sorted by image height with noise to minimize padding while adding randomness.
    
    Groups similar-sized images together to reduce wasted GPU memory from padding,
    while adding noise to prevent strict sorting bias. This is crucial for datasets
    with highly variable image sizes (e.g., phone wireframes 1000-3000px).
    
    Example with batch_size=16:
    - Without sorting: [1024, 1024, ..., 1024, 3000] → all padded to 3000
    - With SmartBatching: [1024±100, 1024±100, ..., 3000±100, 3000±100]
      → groups similar sizes, reduces padding by ~80-95%
    """
    
    def __init__(self, dataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Scan image heights for intelligent batching
        self.heights = self._scan_heights()
    
    def _scan_heights(self):
        """Scan all images to get their heights (cached for efficiency)."""
        heights = []
        logging.info("Scanning dataset heights for SmartBatching (may take a few seconds)...")
        for filename in self.dataset.files:
            try:
                img_path = os.path.join(self.dataset.root_dir, filename)
                with Image.open(img_path) as img:
                    h = img.height
                    # Apply max_height crop if needed
                    if h > self.dataset.max_height:
                        h = self.dataset.max_height
                    heights.append(h)
            except Exception as e:
                logging.warning(f"Failed to scan height for {filename}: {e}")
                heights.append(self.dataset.max_height)
        logging.info(f"Scan complete: {len(heights)} images, height range: {min(heights)}-{max(heights)}px")
        return heights
    
    def __iter__(self):
        indices = np.arange(len(self.dataset))
        
        if self.shuffle:
            # Add ±100px noise to heights to avoid strict sorting (which could bias learning)
            # but keep similar sizes grouped together
            noisy_heights = np.array(self.heights) + np.random.uniform(-100, 100, size=len(indices))
            indices = indices[np.argsort(noisy_heights)]
        else:
            # Pure sorting for reproducibility (validation)
            indices = indices[np.argsort(self.heights)]
        
        # Create batches from sorted indices
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        
        if self.shuffle:
            # Shuffle batch ORDER but keep batch content homogeneous in size
            np.random.shuffle(batches)
        
        for batch in batches:
            yield batch.tolist()
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class VariableSizeDataset(Dataset):
    def __init__(self, 
        root_dir: str,
        noise_level: float = 0.0,
        max_height: int = 2048,
        augment: bool = False,
        files_list: Optional[List[str]] = None,
        sp_prob: float = 0.02,
        perspective_p: float = 0.3,
        perspective_distortion_scale: float = 0.08,
        random_erasing_prob: float = 0.5,
        rotation_degrees: float = 0,
        brightness_jitter: float = 0.0,
        contrast_jitter: float = 0.0
    ) -> None:
        self.root_dir = root_dir
        # Allow passing an explicit files list (used when splitting train/valid)
        if files_list is not None:
            self.files = list(files_list)
        else:
            self.files = [f for f in os.listdir(root_dir) if f.endswith('_linear.png')]
            if len(self.files) == 0:
                self.files = [f for f in os.listdir(root_dir) if f.lower().endswith('.png')]

        self.noise_level = float(noise_level)
        self.max_height = int(max_height) # Hauteur max autorisée (ex: 2048 px)
        self.augment = bool(augment)
        self.sp_prob = float(sp_prob)

        # Read augmentation parameters (defaults kept for compatibility)
        self.perspective_p = float(perspective_p)
        self.perspective_distortion_scale = float(perspective_distortion_scale)
        self.random_erasing_prob = float(random_erasing_prob)
        
        # AMÉLIORATION: Nouvelles augmentations
        self.rotation_degrees = float(rotation_degrees)
        self.brightness_jitter = float(brightness_jitter)
        self.contrast_jitter = float(contrast_jitter)

        # Pipeline d'augmentation "Web-Safe" (appliquée sur PIL images)
        augment_transforms = []
        
        # Rotation légère (si activée)
        if self.rotation_degrees > 0:
            augment_transforms.append(
                T.RandomRotation(degrees=self.rotation_degrees, fill=1.0)  # fill=1 (blanc)
            )
        
        # Perspective
        augment_transforms.append(
            T.RandomPerspective(distortion_scale=self.perspective_distortion_scale, p=self.perspective_p)
        )
        
        # Jitter couleur (brightness + contrast)
        if self.brightness_jitter > 0 or self.contrast_jitter > 0:
            augment_transforms.append(
                T.ColorJitter(brightness=self.brightness_jitter, contrast=self.contrast_jitter)
            )
        
        self.augment_transform = T.Compose(augment_transforms) if augment_transforms else None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if not (0 <= idx < len(self.files)):
            raise IndexError(f"Index {idx} out of range [0, {len(self.files)})")
        
        img_path = os.path.join(self.root_dir, self.files[idx])
        
        # Validation
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        try:
            clean_image = Image.open(img_path).convert('L')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")
        
        # === CROP pour éviter le OOM ===
        w, h = clean_image.size
        if h > self.max_height:
            # Train: crop aléatoire (regularization)
            # Valid: crop déterministe (stabilité du monitoring / scheduler)
            if self.augment:
                top = random.randint(0, h - self.max_height)
            else:
                top = (h - self.max_height) // 2
            # crop(left, top, right, bottom)
            clean_image = clean_image.crop((0, top, w, top + self.max_height))
        # ======================================
        
        # === 2. DATA AUGMENTATION (Seulement pour le train) ===
        if self.augment and self.augment_transform is not None:
            try:
                clean_image = self.augment_transform(clean_image)
            except Exception:
                logging.debug("augment_transform failed for %s", img_path)

        clean_tensor = TF.to_tensor(clean_image)

        # === 3. BRUIT GAUSSIEN (Denoising) ===
        if self.noise_level > 0.0:
            noise = torch.randn_like(clean_tensor) * self.noise_level
            noisy_tensor = clean_tensor + noise
            noisy_tensor = torch.clamp(noisy_tensor, 0.0, 1.0)
        else:
            noisy_tensor = clean_tensor.clone()

        # === 4. RANDOM ERASING (Appliqué SUR L'INPUT pour inpainting robustness) ===
        # We apply RandomErasing to the noisy input only (recommended).
        if self.augment and random.random() < self.random_erasing_prob:
            try:
                eraser = T.RandomErasing(p=1.0, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)
                noisy_tensor = eraser(noisy_tensor)
            except Exception:
                logging.debug("RandomErasing failed for %s", img_path)

        # === 5. SALT-AND-PEPPER (Poivre & Sel) ===
        if self.augment and self.sp_prob > 0.0 and random.random() < 0.5:
            try:
                mask = torch.rand_like(noisy_tensor) < self.sp_prob
                rnd = torch.rand_like(noisy_tensor)
                noisy_tensor = noisy_tensor.clone()
                noisy_tensor[mask & (rnd < 0.5)] = 0.0
                noisy_tensor[mask & (rnd >= 0.5)] = 1.0
            except Exception:
                logging.debug("Salt-and-pepper failed for %s", img_path)

        return noisy_tensor, clean_tensor

def padded_masked_collate(batch):
    """
    Assemble un batch d'images variables avec padding et masque binaire.
    Retourne: (padded_images, padded_targets, masks)
    """
    # batch est une liste de tuples (noisy, clean) retournés par __getitem__
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # 1. Trouver les dimensions max du batch
    max_h = max([img.shape[1] for img in inputs])
    max_w = max([img.shape[2] for img in inputs])

    # Arrondir au multiple de 32 supérieur (pour le VAE qui divise par 32)
    stride = 32
    max_h = ((max_h + stride - 1) // stride) * stride
    max_w = ((max_w + stride - 1) // stride) * stride

    B = len(batch)
    # Tenseurs remplis de zéros (padding par défaut)
    padded_inputs = torch.zeros(B, 1, max_h, max_w)
    padded_targets = torch.zeros(B, 1, max_h, max_w)
    masks = torch.zeros(B, 1, max_h, max_w)

    for i in range(B):
        h, w = inputs[i].shape[1], inputs[i].shape[2]
        
        # Copier l'image en haut à gauche
        padded_inputs[i, :, :h, :w] = inputs[i]
        padded_targets[i, :, :h, :w] = targets[i]
        
        # Créer le masque (1 = pixel valide, 0 = padding)
        masks[i, :, :h, :w] = 1.0

    return padded_inputs, padded_targets, masks


def get_dataloaders(data_config, use_cuda):
    noise = float(data_config.get("noise_level", 0.0))
    # On définit une limite de sécurité (2048 pixels de haut est suffisant pour apprendre les patterns)
    max_h = int(data_config.get("max_height", 2048))
    data_dir = data_config.get('data_dir', './')
    files = [f for f in os.listdir(data_dir) if f.endswith('_linear.png')]
    if len(files) == 0:
        files = [f for f in os.listdir(data_dir) if f.lower().endswith('.png')]
    if len(files) == 0:
        raise ValueError(f"No PNG files found in data_dir={data_dir}")

    # Shuffle and split files into train/valid so we can instantiate datasets
    valid_ratio = float(data_config.get("valid_ratio", 0.2))
    # Reproducible split: allow overriding via `seed` in config
    seed = int(data_config.get('seed', 42))
    rng = random.Random(seed)
    rng.shuffle(files)
    train_size = int((1.0 - valid_ratio) * len(files))
    train_files = files[:train_size]
    valid_files = files[train_size:]

    # Read augmentation params from config (with defaults)
    augment_flag = bool(data_config.get('augment', True))
    sp_prob = float(data_config.get('sp_prob', 0.02))
    random_erasing_prob = float(data_config.get('random_erasing_prob', data_config.get('random_erasing_p', 0.5)))
    perspective_p = float(data_config.get('perspective_p', data_config.get('perspective_p', 0.3)))
    perspective_distortion_scale = float(data_config.get('perspective_distortion_scale', 0.08))
    
    # AMÉLIORATION: Nouvelles augmentations
    rotation_degrees = float(data_config.get('rotation_degrees', 0))
    brightness_jitter = float(data_config.get('brightness_jitter', 0.0))
    contrast_jitter = float(data_config.get('contrast_jitter', 0.0))

    # Create dataset instances with augment enabled for train and disabled for validation
    train_dataset = VariableSizeDataset(
        root_dir=data_dir,
        noise_level=noise,
        max_height=max_h,
        augment=augment_flag,
        files_list=train_files,
        sp_prob=sp_prob,
        perspective_p=perspective_p,
        perspective_distortion_scale=perspective_distortion_scale,
        random_erasing_prob=random_erasing_prob,
        rotation_degrees=rotation_degrees,
        brightness_jitter=brightness_jitter,
        contrast_jitter=contrast_jitter,
    )
    valid_dataset = VariableSizeDataset(
        root_dir=data_dir,
        noise_level=0.0,
        max_height=max_h,
        augment=False,
        files_list=valid_files,
        sp_prob=0.0,
        perspective_p=perspective_p,
        perspective_distortion_scale=perspective_distortion_scale,
        random_erasing_prob=0.0,
        rotation_degrees=0,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
    )

    def _seed_worker(worker_id: int):
        # Ensure python `random` is different per worker
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)

    batch_size = int(data_config.get('batch_size', 1))
    num_workers = int(data_config.get('num_workers', 0))
    g = torch.Generator()
    g.manual_seed(seed)

    # Use SmartBatchSampler for training to minimize padding overhead
    train_sampler = SmartBatchSampler(
        train_dataset,
        batch_size=batch_size,
        shuffle=True  # Shuffle with noise to keep sizes grouped
    )
    
    # For validation: use standard DataLoader (no SmartBatchSampler)
    # This ensures consistent image ordering for reconstruction previews
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=use_cuda,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g,
        collate_fn=padded_masked_collate,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        collate_fn=padded_masked_collate,
    )

    # Best-effort input shape (C, H, W) for model factory + torchinfo.
    # Note: variable-size datasets won't have a single fixed shape.
    try:
        sample_x, _ = train_dataset[0]
        input_size = tuple(sample_x.shape)
    except Exception:
        input_size = (1, max_h, max_h)

    num_classes = int(data_config.get('num_classes', 0))
    return train_loader, valid_loader, input_size, num_classes