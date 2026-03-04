"""Dataset and data-loading utilities for variable-size wireframe images."""

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
    """Group images by height (with noise) to minimise padding waste."""
    
    def __init__(self, dataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.heights = self._scan_heights()

    def _scan_heights(self):
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
            noisy_heights = np.array(self.heights) + np.random.uniform(-100, 100, size=len(indices))
            indices = indices[np.argsort(noisy_heights)]
        else:
            indices = indices[np.argsort(self.heights)]
        
        # Create batches from sorted indices
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        
        if self.shuffle:
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
        random_erasing_prob: float = 0.5,
        hflip_p: float = 0.0,
        vflip_p: float = 0.0,
        translate: Optional[tuple] = None,
        random_crop_p: float = 0.0,
        random_crop_scale: Optional[tuple] = None,
        random_crop_ratio: Optional[tuple] = None,
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
        self.max_height = int(max_height)
        self.augment = bool(augment)
        self.sp_prob = float(sp_prob)
        self.random_erasing_prob = float(random_erasing_prob)
        self.hflip_p = float(hflip_p)
        self.vflip_p = float(vflip_p)
        self.translate = tuple(translate) if translate else None
        self.random_crop_p = float(random_crop_p)
        self.random_crop_scale = tuple(random_crop_scale) if random_crop_scale else (0.5, 1.0)
        self.random_crop_ratio = tuple(random_crop_ratio) if random_crop_ratio else (0.75, 1.33)

        augment_transforms = []

        # Flips: layout symmetry, no aliasing
        if self.hflip_p > 0:
            augment_transforms.append(T.RandomHorizontalFlip(p=self.hflip_p))
        if self.vflip_p > 0:
            augment_transforms.append(T.RandomVerticalFlip(p=self.vflip_p))

        # Translation: shift without rotation (degrees=0 enforced)
        if self.translate is not None:
            augment_transforms.append(
                T.RandomAffine(degrees=0, translate=self.translate, fill=1.0)
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
        
        # Crop to max_height if needed
        w, h = clean_image.size
        if h > self.max_height:
            # Random crop for training (regularization), center crop for validation (reproducibility)
            if self.augment:
                top = random.randint(0, h - self.max_height)
            else:
                top = (h - self.max_height) // 2
            clean_image = clean_image.crop((0, top, w, top + self.max_height))
        
        # Apply augmentation
        if self.augment and self.augment_transform is not None:
            try:
                clean_image = self.augment_transform(clean_image)
            except Exception:
                logging.debug("augment_transform failed for %s", img_path)

        # Random resized crop with NEAREST interpolation (preserves discrete pixel values)
        if self.augment and self.random_crop_p > 0 and random.random() < self.random_crop_p:
            try:
                crop_transform = T.RandomResizedCrop(
                    size=clean_image.size[::-1],  # (H, W)
                    scale=self.random_crop_scale,
                    ratio=self.random_crop_ratio,
                    interpolation=T.InterpolationMode.NEAREST,
                )
                clean_image = crop_transform(clean_image)
            except Exception:
                logging.debug("RandomResizedCrop failed for %s", img_path)

        clean_tensor = TF.to_tensor(clean_image)

        # Add Gaussian noise
        if self.augment and self.noise_level > 0.0:
            noise = torch.randn_like(clean_tensor) * self.noise_level
            noisy_tensor = clean_tensor + noise
            noisy_tensor = torch.clamp(noisy_tensor, 0.0, 1.0)
        else:
            noisy_tensor = clean_tensor.clone()

        # Random erasing
        if self.augment and random.random() < self.random_erasing_prob:
            try:
                eraser = T.RandomErasing(p=1.0, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)
                noisy_tensor = eraser(noisy_tensor)
            except Exception:
                logging.debug("RandomErasing failed for %s", img_path)

        # Salt-and-pepper noise
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
    """Collate variable-size images into a padded batch with binary masks."""
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Find maximum dimensions in batch
    max_h = max([img.shape[1] for img in inputs])
    max_w = max([img.shape[2] for img in inputs])

    # Pad to nearest multiple of 32 (required by VAE architecture)
    stride = 32
    max_h = ((max_h + stride - 1) // stride) * stride
    max_w = ((max_w + stride - 1) // stride) * stride

    B = len(batch)
    # Zero-filled tensors (default padding)
    padded_inputs = torch.zeros(B, 1, max_h, max_w)
    padded_targets = torch.zeros(B, 1, max_h, max_w)
    masks = torch.zeros(B, 1, max_h, max_w)

    for i in range(B):
        h, w = inputs[i].shape[1], inputs[i].shape[2]
        
        # Copy image to top-left corner of padded tensor
        padded_inputs[i, :, :h, :w] = inputs[i]
        padded_targets[i, :, :h, :w] = targets[i]
        
        # Create mask (1 = valid pixel, 0 = padding)
        masks[i, :, :h, :w] = 1.0

    return padded_inputs, padded_targets, masks


def get_dataloaders(data_config, use_cuda):
    noise = float(data_config.get("noise_level", 0.0))
    max_h = int(data_config.get("max_height", 2048))
    
    # Support both old (data_dir with split) and new (train_dir/valid_dir) formats
    train_dir = data_config.get('train_dir')
    valid_dir = data_config.get('valid_dir')
    
    if train_dir and valid_dir:
        # New format: separate train/valid directories
        train_files = [f for f in os.listdir(train_dir) if f.endswith('_linear.png')]
        if len(train_files) == 0:
            train_files = [f for f in os.listdir(train_dir) if f.lower().endswith('.png')]
        
        valid_files = [f for f in os.listdir(valid_dir) if f.endswith('_linear.png')]
        if len(valid_files) == 0:
            valid_files = [f for f in os.listdir(valid_dir) if f.lower().endswith('.png')]
        
        if len(train_files) == 0:
            raise ValueError(f"No PNG files found in train_dir={train_dir}")
        if len(valid_files) == 0:
            raise ValueError(f"No PNG files found in valid_dir={valid_dir}")
            
    else:
        raise ValueError(
            "Config must specify 'train_dir' and 'valid_dir' under data section."
        )

    # Read augmentation params from config
    augment_flag = bool(data_config.get('augment', True))
    sp_prob = float(data_config.get('sp_prob', 0.02))
    random_erasing_prob = float(data_config.get('random_erasing_prob', data_config.get('random_erasing_p', 0.5)))
    hflip_p = float(data_config.get('hflip_p', 0.0))
    vflip_p = float(data_config.get('vflip_p', 0.0))
    translate = data_config.get('translate', None)
    if translate is not None:
        translate = tuple(translate)
    random_crop_p = float(data_config.get('random_crop_p', 0.0))
    random_crop_scale = data_config.get('random_crop_scale', None)
    if random_crop_scale is not None:
        random_crop_scale = tuple(random_crop_scale)
    random_crop_ratio = data_config.get('random_crop_ratio', None)
    if random_crop_ratio is not None:
        random_crop_ratio = tuple(random_crop_ratio)

    # Create dataset instances with augment enabled for train and disabled for validation
    train_dataset = VariableSizeDataset(
        root_dir=train_dir,
        noise_level=noise,
        max_height=max_h,
        augment=augment_flag,
        files_list=train_files,
        sp_prob=sp_prob,
        random_erasing_prob=random_erasing_prob,
        hflip_p=hflip_p,
        vflip_p=vflip_p,
        translate=translate,
        random_crop_p=random_crop_p,
        random_crop_scale=random_crop_scale,
        random_crop_ratio=random_crop_ratio,
    )
    valid_dataset = VariableSizeDataset(
        root_dir=valid_dir,
        noise_level=0.0,
        max_height=max_h,
        augment=False,
        files_list=valid_files,
        sp_prob=0.0,
        random_erasing_prob=0.0,
    )

    def _seed_worker(worker_id: int):
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)

    batch_size = int(data_config.get('batch_size', 1))
    num_workers = int(data_config.get('num_workers', 0))
    seed = int(data_config.get('seed', 42))
    g = torch.Generator()
    g.manual_seed(seed)

    train_sampler = SmartBatchSampler(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
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
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        collate_fn=padded_masked_collate,
    )

    # Best-effort input shape for model factory / torchinfo
    try:
        sample_x, _ = train_dataset[0]
        input_size = tuple(sample_x.shape)
    except Exception:
        input_size = (1, max_h, max_h)

    num_classes = int(data_config.get('num_classes', 0))
    return train_loader, valid_loader, input_size, num_classes