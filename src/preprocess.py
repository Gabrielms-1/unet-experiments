import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from concurrent.futures import ProcessPoolExecutor


def process_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    height, width, _ = image.shape
    
    crops_width = height
    
    limits = [
        width//2 - crops_width//2,
        crops_width,
        width - crops_width,
        width//2 + crops_width//2]

    part_one = image[:, :limits[1]]
    part_two = image[:, limits[0]:limits[3]]
    part_three = image[:, limits[2]:width]
    
    return part_one, part_two, part_three


def process_file(file_path):
    dest_dir = file_path.replace("raw", "preprocessed")
    dest_dir = os.path.splitext(dest_dir)[0]
    part_one, part_two, part_three = process_image(file_path)
    base_name = os.path.basename(file_path).replace(".png", "")
    
    dest_one = dest_dir + "_one.png"
    dest_two = dest_dir + "_two.png"
    dest_three = dest_dir + "_three.png"
    
    cv2.imwrite(dest_one, part_one)
    cv2.imwrite(dest_two, part_two)
    cv2.imwrite(dest_three, part_three)

if __name__ == "__main__":
    base_raw_dir = "data/cityviews/raw"
    train_images_dir = os.path.join(base_raw_dir, "train/img")
    train_masks_dir = os.path.join(base_raw_dir, "train/label")
    val_images_dir = os.path.join(base_raw_dir, "val/img")
    val_masks_dir = os.path.join(base_raw_dir, "val/label")

    train_images = glob.glob(os.path.join(train_images_dir, "*.png"))
    train_masks = glob.glob(os.path.join(train_masks_dir, "*.png"))
    val_images = glob.glob(os.path.join(val_images_dir, "*.png"))
    val_masks = glob.glob(os.path.join(val_masks_dir, "*.png"))

    all_images = train_images + val_images
    all_masks = train_masks + val_masks

    preprocessed_train_images_dir = train_images_dir.replace("cityviews/raw", "cityviews/preprocessed")
    preprocessed_train_masks_dir = train_masks_dir.replace("cityviews/raw", "cityviews/preprocessed")
    preprocessed_val_images_dir = val_images_dir.replace("cityviews/raw", "cityviews/preprocessed")
    preprocessed_val_masks_dir = val_masks_dir.replace("cityviews/raw", "cityviews/preprocessed")

    os.makedirs(preprocessed_train_images_dir, exist_ok=True)
    os.makedirs(preprocessed_train_masks_dir, exist_ok=True)
    os.makedirs(preprocessed_val_images_dir, exist_ok=True)
    os.makedirs(preprocessed_val_masks_dir, exist_ok=True)

    tasks = []
    with ProcessPoolExecutor(max_workers=1) as executor:
        for file_path in all_images:
            tasks.append(executor.submit(process_file, file_path))
        for file_path in all_masks:
            tasks.append(executor.submit(process_file, file_path))
        for task in tasks:
            task.result()
