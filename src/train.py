from unet import UNet
from custom_dataset import CustomDataset
import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from config import *
import argparse
import wandb

def train():
    pass

def validate():
    pass

def main(args):
    wandb.init(project=wandb_project, config={
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_classes": args.num_classes
    })
    
    train_dataset = CustomDataset(
        image_dir=os.path.join(root_dir, "train"),
        mask_dir=os.path.join(root_dir, "train_labels"),
        class_mapping_path=os.path.join(root_dir, "class_mapping.json")
    )

    val_dataset = CustomDataset(
        image_dir=os.path.join(root_dir, "val"),
        mask_dir=os.path.join(root_dir, "val_labels"),
        class_mapping_path=os.path.join(root_dir, "class_mapping.json")
    )

    checkpoint_dir = os.path.join(checkpoint_dir, current_time)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = UNet(3, args.num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=epochs)   
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--learning_rate", type=float, default=learning_rate)
    args = parser.parse_args()

    main(args)