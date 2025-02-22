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
    
    left_image = image[:, :height, :]
    right_image = image[:, width-height:, :]
    
    return right_image, left_image


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
    base_raw_dir = "data/camvid/raw"
    

    