from unet import UNet
from custom_dataset import CustomDataset
import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import config
import argparse
import wandb




def evaluate(model, val_loader, device, criterion):
    model.eval()

    val_loss = 0
    with torch.no_grad():
        for image, mask in val_loader:
            image = image.to(device)
            mask = mask.to(device)

            output = model(image)

            loss = criterion(output, mask)
            val_loss += loss.item()

def train(train_loader, val_loader, model, device, optimizer, criterion):
    for i in range(wandb.config.epochs):
        # batch_loss = 0
        for image, mask in train_loader:
            image = image.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            output = model(image)

            loss = criterion(output, mask)
            loss.backward()

            optimizer.step()

        evaluate(model, val_loader, device, criterion)

        checkpoint_dir = os.path.join(config.checkpoint_dir, config.current_time)
        os.makedirs(checkpoint_dir, exist_ok=True)

        if i % 10 == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": i+1,
                "loss": loss.item()
            }, os.path.join(checkpoint_dir, f"checkpoint_{i}.pth"))

def main(args):
    wandb.init(project=config.wandb_project, config={
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_classes": config.num_classes
    })
    
    train_dataset = CustomDataset(
        image_dir=os.path.join(config.root_dir, "train"),
        mask_dir=os.path.join(config.root_dir, "train_labels"),
        class_mapping_path=os.path.join(config.root_dir, "class_mapping.json")
    )

    val_dataset = CustomDataset(
        image_dir=os.path.join(config.root_dir, "val"),
        mask_dir=os.path.join(config.root_dir, "val_labels"),
        class_mapping_path=os.path.join(config.root_dir, "class_mapping.json")
    )

    checkpoint_dir = os.path.join(config.checkpoint_dir, config.current_time)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = UNet(3, config.num_classes)
    device = torch.device(config.device)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    train(train_loader, val_loader, model, device, optimizer, criterion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.epochs)   
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--learning_rate", type=float, default=config.learning_rate)
    args = parser.parse_args()

    main(args)