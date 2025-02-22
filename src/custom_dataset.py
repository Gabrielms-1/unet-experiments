from torch.utils.data import Dataset
from PIL import Image
from unet import image_transform, mask_transform

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        
    def __len__(self):
        return len(self.image_dir)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_dir[idx])
        mask = Image.open(self.mask_dir[idx])   
        
        image = image_transform(image)
        mask = mask_transform(mask)
            
        return image, mask