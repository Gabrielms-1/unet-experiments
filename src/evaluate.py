import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from unet import UNet
from custom_dataset import CustomDataset
import config
import argparse
import os
import numpy as np
import wandb

def calculate_metrics(preds, targets, num_classes):
    smooth = 1e-10
    iou_per_class = []
    dice_per_class = []
    accuracy_per_class = []
    precision_per_class = []
    recall_per_class = []

    for class_id in range(num_classes):
        pred_class = (preds == class_id)
        target_class = (targets == class_id)

        intersection = (pred_class & target_class).sum()
        union = (pred_class | target_class).sum()
        total = target_class.sum()

        iou = (intersection + smooth) / (union + smooth)
        dice = (2 * intersection + smooth) / (pred_class.sum() + target_class.sum() + smooth)
        accuracy = (intersection + smooth) / (total + smooth)
        precision = (intersection + smooth) / (pred_class.sum() + smooth)
        recall = (intersection + smooth) / (total + smooth)

        iou_per_class.append(iou.item())
        dice_per_class.append(dice.item())
        accuracy_per_class.append(accuracy.item())
        precision_per_class.append(precision.item())
        recall_per_class.append(recall.item())

    return {
        'mean_iou': np.mean(iou_per_class),
        'mean_dice': np.mean(dice_per_class),
        'mean_pixel_accuracy': np.mean(accuracy_per_class),
        'mean_precision': np.mean(precision_per_class),
        'mean_recall': np.mean(recall_per_class),
        'per_class_iou': iou_per_class
    }

def evaluate_model(model, test_loader, device, num_classes, num_examples=10):
    model.eval()
    metrics = {
        'mean_iou': 0,
        'mean_dice': 0,
        'mean_pixel_accuracy': 0,
        'mean_precision': 0,
        'mean_recall': 0,
        'per_class_iou': [0] * num_classes
    }
    example_count = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            batch_metrics = calculate_metrics(preds, masks, num_classes)
            
            for key in metrics:
                if key == 'per_class_iou':
                    for i in range(num_classes):
                        metrics[key][i] += batch_metrics[key][i] / len(test_loader)
                else:
                    metrics[key] += batch_metrics[key] / len(test_loader)

            if example_count < num_examples:
                class_colors = test_loader.dataset.get_class_colors()
                
                rgb_images = (images.cpu() * 255).byte().permute(0, 2, 3, 1)
                gt_masks = apply_colormap(masks.cpu(), class_colors)
                pred_masks = apply_colormap(preds.cpu(), class_colors)
                
                for i in range(min(images.size(0), num_examples - example_count)):
                    wandb.log({
                        "examples": [
                            wandb.Image(rgb_images[i], caption="Original Image"),
                            wandb.Image(gt_masks[i], caption="Ground Truth"),
                            wandb.Image(pred_masks[i], caption="Prediction")
                        ]
                    })
                    example_count += 1

    return metrics

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def apply_colormap(mask_tensor, class_colors):
    colormap = torch.tensor(class_colors, device=mask_tensor.device)
    return colormap[mask_tensor.long()].permute(0, 3, 1, 2)

def main(args):
    wandb.init(
        project=config.wandb_project,
        name=f"eval_{os.path.basename(args.checkpoint_path)}",
        config={
            "checkpoint": args.checkpoint_path,
            "batch_size": args.batch_size
        }
    )

    test_dataset = CustomDataset(
        image_dir=os.path.join(config.root_dir, "test"),
        mask_dir=os.path.join(config.root_dir, "test_labels"),
        class_mapping_path=os.path.join(config.root_dir, "class_mapping.json")
    )

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = UNet(3, config.num_classes).to(config.device)
    model = load_checkpoint(model, args.checkpoint_path)

    metrics = evaluate_model(model, test_loader, config.device, config.num_classes)

    wandb.log({
        "metrics/mean_iou": metrics['mean_iou'],
        "metrics/mean_dice": metrics['mean_dice'],
        "metrics/mean_pixel_accuracy": metrics['mean_pixel_accuracy'],
        "metrics/mean_precision": metrics['mean_precision'],
        "metrics/mean_recall": metrics['mean_recall']
    })
    
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"Mean Dice: {metrics['mean_dice']:.4f}")
    print(f"Pixel Accuracy: {metrics['mean_pixel_accuracy']:.4f}")
    print(f"Precision: {metrics['mean_precision']:.4f}")
    print(f"Recall: {metrics['mean_recall']:.4f}")
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="data/model/checkpoints/2025-02-23_07-29-04/checkpoint_10.pth")
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    args = parser.parse_args()

    main(args)
