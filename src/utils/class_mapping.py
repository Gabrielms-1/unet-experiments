import json
import numpy as np
import glob
import pandas as pd


if __name__ == '__main__':
    mask_dir = 'data/camvid/*_labels/*.png'
    masks = glob.glob(mask_dir)
    
    labels = pd.read_csv('data/camvid/class_dict.csv')
    class_mapping = {"Void": [0, 0, 0]}

    for index, row in labels.iterrows():
        class_name = row['name']
        class_color = [row['r'], row['g'], row['b']]
        class_mapping[class_name] = class_color
    
    with open('data/camvid/class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=4)
