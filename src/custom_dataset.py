from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms
import json
import numpy as np
import torch

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, class_mapping_path):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.masks = [f.replace('right.png', 'L_right.png').replace('left.png', 'L_left.png') for f in self.images]
        
        with open(class_mapping_path) as f:
            class_mapping = json.load(f)
            self.class_mapping = {class_name: tuple(rgb) for class_name, rgb in class_mapping.items()}
        self.class_to_idx = {rgb: i for i, rgb in enumerate(self.class_mapping.values())}
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.images[idx]))
        image = image_transform(image)
        mask = Image.open(os.path.join(self.mask_dir, self.masks[idx]))
        mask = mask.resize((256, 256), Image.NEAREST)
        mask = np.array(mask)
        mask = self.rgb_to_class(mask)
        mask = torch.tensor(mask, dtype=torch.long)
        return image, mask
    
    def get_class_mapping(self):
        return self.class_mapping

    def rgb_to_class(self, mask):
        class_mask = np.zeros((mask.shape[0], mask.shape[1]))
        for _, class_color in self.class_mapping.items():
            idx = self.class_to_idx[class_color]
            class_mask[np.all(mask == class_color, axis=-1)] = idx
        
        return class_mask

    def get_class_colors(self):
        return [v['color'] for v in self.class_info.values()]

image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == "__main__":
    dataset = CustomDataset(
        image_dir="data/camvid/preprocessed/train",
        mask_dir="data/camvid/preprocessed/train_labels",
        class_mapping_path="data/camvid/preprocessed/class_mapping.json"
    )
    
    img, mask = dataset[55]
    print(f"Image shape: {img.shape}")  # Deve ser [3, H, W]
    print(f"Mask shape: {mask.shape}")  # Deve ser [H, W]
    print(f"Unique values na mask: {torch.unique(mask)}")  # Deve incluir 0 e outros índices válidos

