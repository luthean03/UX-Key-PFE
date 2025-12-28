# src/torchtmpl/data.py
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms.functional as TF
import random

Image.MAX_IMAGE_PIXELS = None 

class VariableSizeDataset(Dataset):
    def __init__(self, root_dir, noise_level=0.0, max_height=2048):
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir) if f.endswith('_linear.png')]
        if len(self.files) == 0:
            self.files = [f for f in os.listdir(root_dir) if f.lower().endswith('.png')]
        
        self.noise_level = float(noise_level)
        self.max_height = max_height # Hauteur max autorisée (ex: 2048 px)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.files[idx])
        clean_image = Image.open(img_path).convert('L')
        
        # === RANDOM CROP pour éviter le OOM ===
        w, h = clean_image.size
        if h > self.max_height:
            # On prend une tranche aléatoire de l'interface
            top = random.randint(0, h - self.max_height)
            # crop(left, top, right, bottom)
            clean_image = clean_image.crop((0, top, w, top + self.max_height))
        # ======================================
        
        clean_tensor = TF.to_tensor(clean_image)

        # Ajout du bruit
        if self.noise_level > 0.0:
            noise = torch.randn_like(clean_tensor) * self.noise_level
            noisy_tensor = clean_tensor + noise
            noisy_tensor = torch.clamp(noisy_tensor, 0.0, 1.0)
        else:
            noisy_tensor = clean_tensor.clone()

        return noisy_tensor, clean_tensor

def get_dataloaders(data_config, use_cuda):
    noise = float(data_config.get("noise_level", 0.0))
    # On définit une limite de sécurité (2048 pixels de haut est suffisant pour apprendre les patterns)
    max_h = int(data_config.get("max_height", 2048))
    
    full_dataset = VariableSizeDataset(
        root_dir=data_config.get('data_dir', './'), 
        noise_level=noise,
        max_height=max_h
    )

    train_size = int((1.0 - data_config.get("valid_ratio", 0.2)) * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=use_cuda)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=use_cuda)

    return train_loader, valid_loader, (1, 0, 0), 0