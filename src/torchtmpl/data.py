# src/torchtmpl/data.py
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random
import logging

Image.MAX_IMAGE_PIXELS = None 

class VariableSizeDataset(Dataset):
    def __init__(self, root_dir, noise_level=0.0, max_height=2048, augment=False, files_list=None, sp_prob=0.02, perspective_p=0.3, perspective_distortion_scale=0.08, random_erasing_prob=0.5):
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

        # Pipeline d'augmentation "Web-Safe" (appliquée sur PIL images)
        self.augment_transform = T.Compose([
            T.RandomPerspective(distortion_scale=self.perspective_distortion_scale, p=self.perspective_p),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.files[idx])
        clean_image = Image.open(img_path).convert('L')
        
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
        if self.augment:
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
    )

    def _seed_worker(worker_id: int):
        # Ensure python `random` is different per worker
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)

    batch_size = int(data_config.get('batch_size', 1))
    num_workers = int(data_config.get('num_workers', 0))
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
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