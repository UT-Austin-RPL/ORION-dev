import h5py
from PIL import Image
import argparse
import os
import json
import numpy as np
import cv2
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-folder', required=True)
    parser.add_argument('--mask-val', required=True)
    args = parser.parse_args()
    
    annotations_folder = args.annotation_folder
    mask_val = int(args.mask_val)
    assert(os.path.exists(annotations_folder)), f"Annotation folder {annotations_folder} does not exist."
    config_dict = ""
    assert(os.path.exists(annotations_folder)), f"Annotation folder {annotations_folder} does not exist."
    with open(os.path.join(annotations_folder, "config.json"), "r") as f:
        config_dict = json.load(f)
    
    demo_file = h5py.File(config_dict["original_file"], "r")


    masks = np.load(os.path.join(annotations_folder, "masks.npz"))["arr_0"]
    #change all values of mask to 1 if they are not equal to mask_val
    masks = np.where(masks==mask_val, 0, 1)
    rgb = demo_file.get("data/human_demo/obs/agentview_rgb")[()]

    #create rgb output folder
    output_folder = os.path.join(annotations_folder, "rgb_images")
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for i in range(rgb.shape[0]):
        img = rgb[i]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_folder, f"image{i}.png"), img)
    
    #create mask output folder
    output_folder = os.path.join(annotations_folder, "mask_images")
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for i in range(masks.shape[0]):
        img = masks[i]
        cv2.imwrite(os.path.join(output_folder, f"image{i}.png.png"), img)
    




if __name__ == "__main__":
    main()