import h5py
import os
import numpy as np
import json
import shutil
import cv2
from PIL import Image
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_folder", help="Path to the annotation folder")
    return parser.parse_args()

def create_foundationpose_formatting(annotation_folder):
    # Get the mask/rgb info
    config_dict = {}
    with open(os.path.join(annotation_folder, "config.json"), "r") as f:
        config_dict = json.load(f)

    has_depth = False
    with h5py.File(config_dict["original_file"], "r") as demo_file:
        rgb = demo_file["data/human_demo/obs/agentview_rgb/"][()]
        if "data/human_demo/obs/agentview_depth/" in demo_file:
            has_depth = True
            depth = demo_file["data/human_demo/obs/agentview_depth/"][()]
        else:
            depth = np.zeros_like(rgb)
        data_config = json.loads(demo_file["data"].attrs['data_config'])
        K = data_config["intrinsics"]["camera_rs_0"]
        
    masks = np.load(os.path.join(annotation_folder, "masks.npz"))['arr_0']
    mask_indices = sorted(np.unique(masks))[1:] # dont use 0 because it is the background

    all_ob_masks = []

    for i in range(len(mask_indices)):
        ob_specific_mask = masks.copy()
        ob_specific_mask[ob_specific_mask != mask_indices[i]] = 0
        ob_specific_mask[ob_specific_mask == mask_indices[i]] = 255
        
        all_ob_masks.append(ob_specific_mask)
    
    # Write to FoundationPose format
    foundation_pose_dir = os.path.join(annotation_folder, "FoundationPose")
    os.makedirs(foundation_pose_dir, exist_ok=True)

    rgb_dir = os.path.join(foundation_pose_dir, "rgb")
    if os.path.exists(rgb_dir):
        shutil.rmtree(rgb_dir)
    os.makedirs(rgb_dir)

    for i in range(rgb.shape[0]):
        Image.fromarray(rgb[i]).save(os.path.join(rgb_dir, f"rgb_{i:04d}.png"))

    depth_dir = os.path.join(foundation_pose_dir, "depth")
    if os.path.exists(depth_dir):
        shutil.rmtree(depth_dir)
    os.makedirs(depth_dir)
    for i in range(depth.shape[0]):
        cv2.imwrite(os.path.join(depth_dir, f"depth_{i:04d}.png"), depth[i])

    for i, mask_index in enumerate(mask_indices):
        mask_dir = os.path.join(foundation_pose_dir, f"ob_{mask_index}_masks")
        if os.path.exists(mask_dir):
            shutil.rmtree(mask_dir)
        os.makedirs(mask_dir)
        for j in range(all_ob_masks[i].shape[0]):
            cv2.imwrite(os.path.join(mask_dir, f"ob_{mask_index}_masks_{j:04d}.png"), all_ob_masks[i][j])

    # write intrinsics
    K = np.zeros((3, 3))
    K[0,0] = data_config["intrinsics"]["camera_rs_0"]["fx"]
    K[1,1] = data_config["intrinsics"]["camera_rs_0"]["fy"]
    K[0,2] = data_config["intrinsics"]["camera_rs_0"]["cx"]
    K[1,2] = data_config["intrinsics"]["camera_rs_0"]["cy"]
    K[2,2] = 1

    #write k to txt
    np.savetxt(os.path.join(foundation_pose_dir, "cam_K.txt"), K, fmt='%.6f')
    print(f"{K=}")



if __name__ == "__main__":
    args = parse_args()
    create_foundationpose_formatting(args.annotation_folder)

