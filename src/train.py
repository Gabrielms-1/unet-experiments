from unet import UNet
from custom_dataset import CustomDataset
import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from config import *



def train():
    pass

def validate():
    pass


if __name__ == "__main__":
    
    train_dataset = CustomDataset(
        image_dir=os.path.join(root_dir, "train/img"),
        mask_dir=os.path.join(root_dir, "train/label"),
    )

    val_dataset = CustomDataset(
        image_dir=os.path.join(root_dir, "val/img"),
        mask_dir=os.path.join(root_dir, "val/label"),
    )

    checkpoint_dir = os.path.join(checkpoint_dir, current_time)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = UNet(3, 8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    train()