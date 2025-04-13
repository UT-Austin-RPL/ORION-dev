import h5py
import os
import numpy as np
import json
from PIL import Image
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_folder", help="Path to the annotation folder")
    parser.add_argument("--mask_idx", type=int, help="What mask index to keep in the image", default=1)
    parser.add_argument("--frame_idx", type=int, help="What frame to use", default=0)
    return parser.parse_args()

def create_transparent_bg_from_mask(rgb, mask, mask_idx, save_name=None):
    target_ob = mask_idx
    mask = mask.copy()
    mask[mask != target_ob] = 0
    mask[mask == target_ob] = 255
    mask_target = Image.fromarray(mask).convert('L')
    
    mesh_img_folder = Path("/home/arisrei/research/RPL/ORION/images_for_meshes")

    #recenter the image by cropping
    mask = np.array(mask_target)
    nonzero = np.nonzero(mask)
    
    #padding = np.zeros(2, dtype=int)
    padding_amt = np.max(np.max(nonzero, axis=1) - np.min(nonzero, axis=1)) // 2
    padding = np.array([padding_amt, padding_amt])

    y1 = max(0, np.min(nonzero[0]) - padding[0])
    y2 = min(np.max(nonzero[0]) + padding[0], mask.shape[0])
    x1 = max(0, np.min(nonzero[1]) - padding[1])
    x2 = min(np.max(nonzero[1]) + padding[1], mask.shape[1])

    recentered_mask = Image.fromarray(mask[y1:y2, x1:x2])
    recentered_rgb = Image.fromarray(rgb[y1:y2, x1:x2, :])
    recentered_image_with_transparency = np.dstack((recentered_rgb, recentered_mask))
    full_image_with_transparency = np.dstack((Image.fromarray(rgb), mask_target))
    
    if save_name is not None:    
        os.makedirs(os.path.join(mesh_img_folder, save_name), exist_ok=True)
        Image.fromarray(recentered_image_with_transparency).save(os.path.join(mesh_img_folder, save_name, f"{save_name}_rmbg.png"))
        recentered_rgb.save(os.path.join(mesh_img_folder, save_name, f"{save_name}_with_bg.png"))
        Image.fromarray(full_image_with_transparency).save(os.path.join(mesh_img_folder, save_name, f"{save_name}_full_rmbg.png"))

        print(f"saving to {os.path.join(mesh_img_folder, save_name, f'{save_name}_rmbg.png')}")
        print(f"saving to {os.path.join(mesh_img_folder, save_name, f'{save_name}_with_bg.png')}")
        print(f"saving to {os.path.join(mesh_img_folder, save_name, f'{save_name}_full_rmbg.png')}")
        
    return recentered_image_with_transparency, recentered_rgb, full_image_with_transparency
    
    
    

def create_transparent_bg_from_mask_annotation_folder(annotation_folder, mask_idx, frame_idx):
    config_dict = {}
    with open(os.path.join(annotation_folder, "config.json"), "r") as f:
        config_dict = json.load(f)

    with h5py.File(config_dict["original_file"], "r") as demo_file:
        rgb = demo_file["data/human_demo/obs/agentview_rgb/"][()][frame_idx]

    mask = np.load(os.path.join(annotation_folder, "masks.npz"))['arr_0'][frame_idx]
    img_folder = annotation_folder.split("/")[-1]    
    
    create_transparent_bg_from_mask(rgb, mask, mask_idx, save_name=img_folder)

    



if __name__ == "__main__":
    args = parse_args()
    create_transparent_bg_from_mask(args.annotation_folder, args.mask_idx, args.frame_idx)