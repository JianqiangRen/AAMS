# coding=utf-8
# summary: Code of CVPR 2019 accepted paper Attention-aware Multi-stroke Style Transfer

import glob
import os
from PIL import Image
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", dest='dataset', type=str)
args = parser.parse_args()


if __name__ == "__main__":
    img_paths = glob.glob(os.path.join(args.dataset, "*.*"))
    
    total_count = len(img_paths)
    for i, image_path in enumerate(img_paths):
        img = Image.open(image_path)
        img = np.asarray(img)
        
        if len(np.shape(img)) != 3 or np.shape(img)[2] != 3:
            print(image_path + " fomat illegal[{}/{}]".format(i, total_count))
            os.remove(image_path)
    
    print("clean done")