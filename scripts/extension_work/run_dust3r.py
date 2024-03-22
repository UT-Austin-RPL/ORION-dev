import sys
import os


import numpy as np
import os
import torch
import argparse

from PIL import Image
import cv2
import json
import h5py

from create_dust3r_video import create_video

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-folder", type=str, default="/home/arisrei/research/RPL/ORION/annotations/human_demo/yellow_bottle")
    parser.add_argument("--images_folder", type=str, default="/home/arisrei/research/RPL/ORION/datasets/iphones/yellow_bottle/images")
    parser.add_argument("--dust3r_path", type=str, default="/home/arisrei/research/RPL/ORION/third_party/dust3r")
    args = parser.parse_args()

    annotation_dir = args.annotation_folder
    dust3r_path = args.dust3r_path
    sys.path.append(dust3r_path)

    from dust3r.inference import inference, load_model
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from dust3r.utils.device import to_numpy

    model_path = os.path.join(dust3r_path, "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
    device = 'cuda:1'
    batch_size = 1
    schedule = 'linear'
    lr = 0.01
    niter = 300

    model = load_model(model_path, device)

    num_files = len(os.listdir(args.images_folder))
    file_list = []
    for i in range(40,96):
    #for i in range(num_files):
        file_list.append(os.path.join(args.images_folder, f"image{i}.png"))

    image_batch_size = 28
    overlap_percent = .25
    
    #TODO: Change back to normal when done stress testing
    #image_batch_size = 25
    #overlap_percent = .2
    start_idx = 0
    end_idx = image_batch_size

    depths_unormalized = []
    num_depths = 0
    imgs = []
    extrinsics = []
    intrinsics = []
    pts3d_all = []
    col_all = []

    while start_idx < len(file_list) - int(overlap_percent * image_batch_size):

        #always use the first frame to maintain same relative poses
        if start_idx == 0:
            images = load_images(file_list[start_idx:end_idx], size=512)
        else:
            images = load_images([file_list[0]] + file_list[start_idx:end_idx], size=512)
            
        print("num images:", len(images))

        print(start_idx, end_idx, len(depths_unormalized), num_depths)
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=batch_size)

        mode = GlobalAlignerMode.PointCloudOptimizer if len(images) > 2 else GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=device, mode=mode)
        scene.min_conf_thr = float(scene.conf_trf(torch.tensor(3)))

        if mode == GlobalAlignerMode.PointCloudOptimizer:
            loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

        scene = scene.clean_pointcloud()

        # retrieve useful values from scene:
        imgs_curr = scene.imgs
        pts3d_curr = scene.get_pts3d()
        intrinsics_curr = scene.get_intrinsics().cpu().detach().numpy()
        extrinsics_curr = scene.get_im_poses().cpu().detach().numpy()

        mask = to_numpy(scene.get_masks())
        pts3d_curr = to_numpy(pts3d_curr)
        imgs_curr = np.array(to_numpy(imgs_curr))
        depths_unormalized_curr = np.array(to_numpy(scene.get_depthmaps()))

        pts3d_curr = np.concatenate([p[m] for p, m in zip(pts3d_curr, mask)])
        col_curr = np.concatenate([p[m] for p, m in zip(imgs_curr, mask)])

        #if end_idx >= len(file_list):
        if start_idx == 0:
            print("ADDING WITH SHAPE", depths_unormalized_curr.shape)
            depths_unormalized.append(depths_unormalized_curr)
            imgs.append(imgs_curr)
            intrinsics.append(intrinsics_curr)
            extrinsics.append(extrinsics_curr)
            pts3d_all.append(pts3d_curr)
            col_all.append(col_curr)
            num_depths += depths_unormalized_curr.shape[0]
        else:
            print("ADDING WITH SHAPE", depths_unormalized_curr[1 + int(image_batch_size * overlap_percent):, ...].shape, 1 + int(image_batch_size * overlap_percent))
            depths_unormalized.append(depths_unormalized_curr[1 + int(image_batch_size * overlap_percent):, ...])
            imgs.append(imgs_curr[1 + int(image_batch_size * overlap_percent):, ...])
            intrinsics.append(intrinsics_curr[1 + int(image_batch_size * overlap_percent):])
            extrinsics.append(extrinsics_curr[1 + int(image_batch_size * overlap_percent):])
            num_depths += depths_unormalized_curr[1 + int(image_batch_size * overlap_percent):, ...].shape[0]
            
        start_idx = end_idx - int(overlap_percent * image_batch_size)
        end_idx = start_idx + image_batch_size

    print(start_idx, end_idx, len(depths_unormalized), num_depths)
    depths_unormalized = np.concatenate(depths_unormalized, axis=0)
    imgs = np.concatenate(imgs, axis=0)
    intrinsics = np.concatenate(intrinsics, axis=0)
    extrinsics = np.concatenate(extrinsics, axis=0)

    print(len(file_list), depths_unormalized.shape[0]); assert len(file_list) == depths_unormalized.shape[0]
    print(len(file_list), imgs.shape[0]); assert len(file_list) == imgs.shape[0]

    with open(os.path.join(annotation_dir, "config.json"), "r") as f:
        config_dict = json.load(f)

    demo_file = h5py.File(config_dict["original_file"], "r+")
    if "dust3r_depth" in demo_file["data/human_demo/obs/"]:
        del demo_file["data/human_demo/obs/dust3r_depth"]
    if "dust3r_rgb" in demo_file["data/human_demo/obs/"]:
        del demo_file["data/human_demo/obs/dust3r_rgb"]
    if "dust3r_intrinsics" in demo_file["data/human_demo/obs/"]:
        del demo_file["data/human_demo/obs/dust3r_intrinsics"]
    if "dust3r_extrinsics" in demo_file["data/human_demo/obs/"]:
        del demo_file["data/human_demo/obs/dust3r_extrinsics"]
    if "scene_points" in demo_file["data/human_demo/obs/"]:
        del demo_file["data/human_demo/obs/scene_points"]
    if "scene_col" in demo_file["data/human_demo/obs/"]:
        del demo_file["data/human_demo/obs/scene_col"]

    demo_file.create_dataset("data/human_demo/obs/dust3r_depth", data=depths_unormalized)
    demo_file.create_dataset("data/human_demo/obs/dust3r_rgb", data=imgs)
    demo_file.create_dataset("data/human_demo/obs/dust3r_intrinsics", data=intrinsics)
    demo_file.create_dataset("data/human_demo/obs/dust3r_extrinsics", data=extrinsics)
    demo_file.create_dataset("data/human_demo/obs/scene_points", data=pts3d_all[0])
    demo_file.create_dataset("data/human_demo/obs/scene_col", data=col_all[0])

    demo_file.close()
    # np.save(os.path.join(annotation_dir, "dust3r/depths.npy"), depths_unormalized)
    # np.save(os.path.join(annotation_dir, "dust3r/images.npy"), imgs)

    create_video((imgs * 255).astype(np.uint8), os.path.join(annotation_dir, "dust3r/run_dust3r_segment.mp4"))
    
    # config_dict["video_file_dust3r"] = os.path.join(annotation_dir, "dust3r/run_dust3r_segment.mp4")
    # with open(os.path.join(annotation_dir, "config.json"), "w") as f:
    #     json.dump(config_dict, f)

if __name__ == '__main__':
    main()