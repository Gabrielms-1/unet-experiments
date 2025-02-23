import cv2
import os
import glob
from multiprocessing import Pool

def process_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    height, width, _ = image.shape
    
    left_image = image[:, :height, :]
    right_image = image[:, width-height:, :]
    
    return right_image, left_image


def process_file(file_path):
    
    dest_dir = file_path.replace("raw", "preprocessed")
    right_image, left_image = process_image(file_path)
    
    dest_right = dest_dir.replace(".png", "_right.png")
    dest_left = dest_dir.replace(".png", "_left.png")
    
    os.makedirs(os.path.dirname(dest_dir), exist_ok=True)

    cv2.imwrite(dest_right, right_image)
    cv2.imwrite(dest_left, left_image)

if __name__ == "__main__":
    base_raw_dir = "data/camvid/raw"
    base_preprocessed_dir = "data/camvid/preprocessed"
    
    
    with Pool(processes=3) as pool:
        files = glob.glob(os.path.join(base_raw_dir, "**/*.png"), recursive=True)
        pool.map(process_file, files)
    