import json
import os
import argparse
from pathlib import Path

from PIL import Image
import numpy as np
import h5py
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-folder", default="aircraft_demo")
    parser.add_argument("--tgt-obj", default=1)
    parser.add_argument("--dont-use-transparent", default=False, action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    annotation_folder = args.annotation_folder
    tgt_obj = args.tgt_obj
    dont_use_transparent = args.dont_use_transparent

    img_path = create_transparent_bg(annotation_folder, tgt_obj, dont_use_transparent)
    print(create_mesh_InstantMesh(img_path))

def create_mesh_InstantMesh(img_path: str):
    #activate_command = f"conda init && conda activate InstantMesh"
    #full_command = f"{activate_command} && python third_party/InstantMesh/run.py {img_path} third_party/InstantMesh/configs/instant-mesh-large.yaml --save_video --export_texmap --save_to_mesh_folder"
    #result = subprocess.run(full_command, shell=True, executable="/bin/bash", capture_output=True, text=True)
    # result = subprocess.run(f"""
    #     python third_party/InstantMesh/run.py third_party/InstantMesh/configs/instant-mesh-large.yaml {img_path} --save_video --export_texmap --save_to_mesh_folder
    #                """,
    #     shell=True, executable="/bin/bash", capture_output=True, text=True)
    # #print(result)
    # result.stdout, result.stderr = result.stdout.splitlines(), result.stderr.splitlines()
    #return result.stdout, result.stderr

    cmd = f"cd third_party/InstantMesh && conda run -n instantmesh python run.py configs/instant-mesh-large.yaml {img_path} --save_video --export_texmap --save_to_mesh_folder"
    os.system(cmd)

def create_transparent_bg(annotation_folder: str, target_ob: int, dont_use_transparent: bool):
    config_dict = {}
    with open(os.path.join(annotation_folder, "config.json"), "r") as f:
        config_dict = json.load(f)

    with h5py.File(config_dict["original_file"], "r") as demo_file:
        rgb = demo_file["data/human_demo/obs/agentview_rgb/"][()]
    masks = np.load(os.path.join(annotation_folder, "masks.npz"))['arr_0']
    idx = 0

    masks[masks != target_ob] = 0
    masks[masks == target_ob] = 255

    mask_target = Image.fromarray(masks[idx]).convert('L')

    img_folder = annotation_folder.split("/")[-1]
    mesh_img_folder = Path("/home/arisrei/research/RPL/ORION/images_for_meshes")
    print(f"saving to {os.path.join(mesh_img_folder, img_folder, f'{img_folder}.png')}")
    print(f"saving to {os.path.join(mesh_img_folder, img_folder, f'{img_folder}_with_bg.png')}")

    #recenter the image by cropping
    mask = np.array(mask_target)
    nonzero = np.nonzero(mask)
    padding = np.array(mask.shape)//10
    y1 = max(0, np.min(nonzero[0]) - padding[0])
    y2 = min(np.max(nonzero[0]) + padding[0], mask.shape[0])
    x1 = max(0, np.min(nonzero[1]) - padding[1])
    x2 = min(np.max(nonzero[1]) + padding[1], mask.shape[1])
    print(f"({y1}, {x1}) ({y2}, {x2})")

    recentered_mask = Image.fromarray(mask[y1:y2, x1:x2])
    recentered_rgb = Image.fromarray(rgb[idx][y1:y2, x1:x2, :])

    recentered_image_with_transparency = np.dstack((recentered_rgb, recentered_mask))

    os.makedirs(os.path.join(mesh_img_folder, img_folder), exist_ok=True)
    Image.fromarray(recentered_image_with_transparency).save(os.path.join(mesh_img_folder, img_folder, f"{img_folder}.png"))
    recentered_rgb.save(os.path.join(mesh_img_folder, img_folder, f"{img_folder}_with_bg.png"))

    if dont_use_transparent:
        return os.path.join(mesh_img_folder, img_folder, f'{img_folder}_with_bg.png')
    else:
        return os.path.join(mesh_img_folder, img_folder, f'{img_folder}.png')


if __name__ == "__main__":
    main()