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


    with torch.no_grad():
        epoch_loss = 0
        batch_loss = 0
        for image, mask in val_loader:
            image = image.to(device)
            mask = mask.to(device)

            output = model(image)

            loss = criterion(output, mask)
            batch_loss += loss.item()

            pred = torch.argmax(output, dim=1)


        epoch_loss = batch_loss / len(val_loader)
        

    return epoch_loss

def train(train_loader, val_loader, model, device, optimizer, criterion):
    train_loss = []
    val_loss = []
    
    # epochs:
    for i in range(wandb.config.epochs):
        batch_loss = 0
        # batch
        for image, mask in train_loader:
            image = image.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            output = model(image)

            loss = criterion(output, mask)
            loss.backward()

            optimizer.step()

            batch_loss += loss.item()
        
        epoch_loss = batch_loss / len(train_loader)
        train_loss.append(epoch_loss)
        current_val_loss = evaluate(model, val_loader, device, criterion)
        val_loss.append(current_val_loss)
        
        print("logging in wandb")
        wandb.log({
            "epoch": i+1,
            "train_loss": epoch_loss,
            "val_loss": current_val_loss,
        })

        checkpoint_dir = os.path.join(config.checkpoint_dir, config.current_time)
        os.makedirs(checkpoint_dir, exist_ok=True)

        if i % 10 == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": i+1,
                "train_loss": epoch_loss,
                "val_loss": current_val_loss,
            }, os.path.join(checkpoint_dir, f"checkpoint_{i}.pth"))

        print(f"Epoch {i+1} loss: {epoch_loss}, val_loss: {current_val_loss}")

    return train_loss, val_loss


def main(args):
    wandb.init(
        project=config.wandb_project, 
        name=f"unet_camvid_{config.current_time}",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_classes": config.num_classes
        }
    )
    
    train_dataset = CustomDataset(
        image_dir=os.path.join(args.input_data_dir, "train"),
        mask_dir=os.path.join(args.input_data_dir, "train_labels"),
        class_mapping_path=os.path.join(args.input_data_dir, "class_mapping.json"),
        resize=args.resize
    )

    val_dataset = CustomDataset(
        image_dir=os.path.join(args.input_data_dir, "val"),
        mask_dir=os.path.join(args.input_data_dir, "val_labels"),
        class_mapping_path=os.path.join(args.input_data_dir, "class_mapping.json"),
        resize=args.resize
    )

    checkpoint_dir = os.path.join(args.output_dir, args.current_time)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = UNet(3, config.num_classes)
    device = torch.device(config.device)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    train_loss, val_loss = train(train_loader, val_loader, model, device, optimizer, criterion)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.epochs)   
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--learning_rate", type=float, default=config.learning_rate)
    parser.add_argument("--input_data_dir", type=str, default=config.root_dir)
    parser.add_argument("--output_dir", type=str, default=config.checkpoint_dir)
    parser.add_argument("--resize", type=int, default=config.resize)
    
    args, _ = parser.parse_known_args()
    args.current_time = config.current_time

    main(args)