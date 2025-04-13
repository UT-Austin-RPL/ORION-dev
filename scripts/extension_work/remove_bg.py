from orion.algos.grounded_sam_wrapper import GroundedSamWrapper
import argparse
import os
from pathlib import Path
import cv2
from PIL import Image
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="A simple user interface for XMem")
    parser.add_argument("--image", help="Path to an image file")
    parser.add_argument("--prompt", type=str, help="Prompt for segmentation")
    
    return parser.parse_args()

def main():
    args = parse_args()
    wrapper = GroundedSamWrapper()
    parent_dir = os.path.dirname(args.image)
    out_dir = os.path.join(parent_dir, "masked_images")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    raw_img = Image.open(args.image)
    masked_img = wrapper.segment(cv2.imread(args.image), [args.prompt])
    print(type(masked_img))

    masked_img_np = np.array(masked_img)
    masked_img_np[masked_img_np != 1] = 0
    masked_img_np[masked_img_np == 1] = 255

    image_with_transparency = np.dstack((raw_img, Image.fromarray(masked_img_np).convert('L')))
    Image.fromarray(image_with_transparency).save(os.path.join(out_dir, os.path.basename(args.image).split(".")[0] + "_masked.png"))

if __name__ == "__main__":
    main()
