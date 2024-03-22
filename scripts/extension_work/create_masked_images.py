import h5py
from PIL import Image
import argparse
import os
import json
import numpy as np
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

    #create masked_output folder
    output_folder = os.path.join(annotations_folder, "masked_images")
    masked_rgb = rgb * masks[..., np.newaxis]
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    for i in range(masked_rgb.shape[0]):
        img = masked_rgb[i]
        img = Image.fromarray(img.astype(np.uint8))
        img.save(os.path.join(output_folder, f"image{i}.png"))




if __name__ == "__main__":
    main()