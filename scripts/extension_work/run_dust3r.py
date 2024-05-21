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
import math
from create_dust3r_video import create_video

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-folder", type=str, default="/home/arisrei/research/RPL/ORION/annotations/human_demo/yellow_bottle")
    parser.add_argument("--dust3r-path", type=str, default="/home/arisrei/research/RPL/ORION/third_party/dust3r")
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--start-end", type=int, nargs=2)
    args = parser.parse_args()

    annotation_dir = args.annotation_folder
    dust3r_path = args.dust3r_path
    sys.path.append(dust3r_path)

    from dust3r.inference import inference, load_model
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from dust3r.utils.device import to_numpy

    with open(os.path.join(annotation_dir, "config.json"), "r") as f:
        config_dict = json.load(f)
    images_folder = config_dict["images_folder"]
    print(images_folder)
    num_files = len(os.listdir(images_folder))
    file_list = []
    if args.start_end:
        start = args.start_end[0]
        end = args.start_end[1]

        print(start,end)
        for i in range(start, end):
            file_list.append(os.path.join(images_folder, f"image{i}.png"))
    else:
        for i in range(num_files):
            file_list.append(os.path.join(images_folder, f"image{i}.png"))

    image_batch_size = 25
    overlap_percent = .2
    
    start_idx = 0
    end_idx = image_batch_size

    depths_unormalized = []
    num_depths = 0
    imgs = []
    extrinsics = []
    intrinsics = []
    pts3d_all = []
    col_all = []

    image_dims = load_images(file_list[0:2], size=512)[0]['img'].shape[2:]

    #calculate total number of batches that will be run
    num_batches = math.ceil(len(file_list) / (image_batch_size * (1 - overlap_percent)))

    extrinsic_batches = np.zeros((num_batches, image_batch_size, 4, 4))
    intrinsic_batches = np.zeros((num_batches, image_batch_size, 3, 3))
    pts3d_per_batch = []
    depths_unormalized_per_batch = np.zeros((num_batches, image_batch_size, image_dims[0], image_dims[1]))

    batch_num = 0

    model_path = os.path.join(dust3r_path, "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
    device = 'cuda:' + str(args.cuda_device)
    batch_size = 1
    schedule = 'linear'
    lr = 0.01
    niter = 300

    model = load_model(model_path, device)
    while start_idx == 0 or start_idx < len(file_list) - int(overlap_percent * image_batch_size): #always run at least once

        images = load_images(file_list[start_idx:end_idx], size=512)
            
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
        
        extrinsic_batches[batch_num][0:len(images)] = extrinsics_curr
        intrinsic_batches[batch_num][0:len(images)] = intrinsics_curr
        
        pts3d_per_batch.append(pts3d_curr)
        depths_unormalized_per_batch[batch_num][0:len(images)] = depths_unormalized_curr

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
            print("ADDING WITH SHAPE", depths_unormalized_curr[int(image_batch_size * overlap_percent):, ...].shape, int(image_batch_size * overlap_percent))
            depths_unormalized.append(depths_unormalized_curr[int(image_batch_size * overlap_percent):, ...])
            imgs.append(imgs_curr[int(image_batch_size * overlap_percent):, ...])
            intrinsics.append(intrinsics_curr[int(image_batch_size * overlap_percent):])
            extrinsics.append(extrinsics_curr[int(image_batch_size * overlap_percent):])
            num_depths += depths_unormalized_curr[int(image_batch_size * overlap_percent):, ...].shape[0]
            
        start_idx = end_idx - int(overlap_percent * image_batch_size)
        end_idx = start_idx + image_batch_size
        batch_num += 1

    print(start_idx, end_idx, len(depths_unormalized), num_depths)
    depths_unormalized = np.concatenate(depths_unormalized, axis=0)
    imgs = np.concatenate(imgs, axis=0)
    intrinsics = np.concatenate(intrinsics, axis=0)
    extrinsics = np.concatenate(extrinsics, axis=0)

    print(len(file_list), depths_unormalized.shape[0]); assert len(file_list) == depths_unormalized.shape[0]
    print(len(file_list), imgs.shape[0]); assert len(file_list) == imgs.shape[0]

    transformed_extrinsics_per_batch = np.zeros_like(extrinsic_batches)
    transformed_extrinsics_per_batch[0] = extrinsic_batches[0]
    transforms_per_batch = np.zeros((transformed_extrinsics_per_batch.shape[0], 4, 4)) #stores the transformation matrices for each batch. transforms extrinsics in the batch to match the same coordinate frame as the first batch
    transforms_per_batch[0] = np.eye(4)


    for i in range(1, transformed_extrinsics_per_batch.shape[0]):
        #use the transformed extrinsics from the previous batch since those are already in the same coordinate frame as the first batch
        T_B1 = transformed_extrinsics_per_batch[i-1][image_batch_size - int(image_batch_size * overlap_percent)]
        T_B2 = extrinsic_batches[i][0]
        
        #find matrix that transforms B2 to B1
        T_B2_inv = np.zeros((4, 4))
        T_B2_inv[3,3] = 1
        T_B2_inv[0:3,0:3] = T_B2[0:3,0:3].T
        T_B2_inv[0:3,3] = -1 * T_B2[0:3,0:3].T @ T_B2[0:3,3]        

        T_B1_B2 = T_B1 @ T_B2_inv
        transforms_per_batch[i] = T_B1_B2

        #apply T_B1_B2 to each extrinsic in batch i
        for j in range(transformed_extrinsics_per_batch.shape[1]):
            transformed_extrinsics_per_batch[i][j] = T_B1_B2 @ extrinsic_batches[i][j]

    extrinsics_correct_frame = []
    for i in range(transformed_extrinsics_per_batch.shape[0]):
        if i == 0:
            extrinsics_correct_frame.append(transformed_extrinsics_per_batch[i])
        elif i != transformed_extrinsics_per_batch.shape[0] - 1:
            extrinsics_correct_frame.append(transformed_extrinsics_per_batch[i, int(image_batch_size * overlap_percent):])
        else:
            end = extrinsics.shape[0] % (image_batch_size - int(image_batch_size * overlap_percent))
            extrinsics_correct_frame.append(transformed_extrinsics_per_batch[i, int(image_batch_size * overlap_percent):end])
    extrinsics_correct_frame = np.concatenate(extrinsics_correct_frame, axis=0)
    
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
        
    if "dust3r_extrinsic_batches" in demo_file["data/human_demo/obs/"]:
        del demo_file["data/human_demo/obs/dust3r_extrinsic_batches"]
    if "dust3r_intrinsic_batches" in demo_file["data/human_demo/obs/"]:
        del demo_file["data/human_demo/obs/dust3r_intrinsic_batches"]
    if 'data/human_demo/obs/dust3r_extrinsics_correct_frame' in demo_file:
        del demo_file['data/human_demo/obs/dust3r_extrinsics_correct_frame']
    if 'data/human_demo/obs/dust3r_transforms_per_batch' in demo_file:
        del demo_file['data/human_demo/obs/dust3r_transforms_per_batch']
    if 'data/human_demo/obs/dust3r_depths_unormalized_per_batch' in demo_file:
        del demo_file['data/human_demo/obs/dust3r_depths_unormalized_per_batch']

    demo_file.create_dataset("data/human_demo/obs/dust3r_depth", data=depths_unormalized)
    demo_file.create_dataset("data/human_demo/obs/dust3r_rgb", data=imgs)
    demo_file.create_dataset("data/human_demo/obs/dust3r_intrinsics", data=intrinsics)
    demo_file.create_dataset("data/human_demo/obs/dust3r_extrinsics", data=extrinsics)
    demo_file.create_dataset("data/human_demo/obs/scene_points", data=pts3d_all[0])
    demo_file.create_dataset("data/human_demo/obs/scene_col", data=col_all[0])
    demo_file.create_dataset("data/human_demo/obs/dust3r_extrinsic_batches", data=extrinsic_batches)
    demo_file.create_dataset("data/human_demo/obs/dust3r_intrinsic_batches", data=intrinsic_batches)
    demo_file.create_dataset('data/human_demo/obs/dust3r_transforms_per_batch', data=transforms_per_batch)
    demo_file.create_dataset('data/human_demo/obs/dust3r_depths_unormalized_per_batch', data=depths_unormalized_per_batch)
    demo_file.create_dataset('data/human_demo/obs/dust3r_extrinsics_correct_frame', data=extrinsics_correct_frame)

    demo_file.close()
    # np.save(os.path.join(annotation_dir, "dust3r/depths.npy"), depths_unormalized)
    # np.save(os.path.join(annotation_dir, "dust3r/images.npy"), imgs)

    create_video((imgs * 255).astype(np.uint8), os.path.join(annotation_dir, "dust3r/run_dust3r_segment.mp4"))
    
    # config_dict["video_file_dust3r"] = os.path.join(annotation_dir,
    # "dust3r/run_dust3r_segment.mp4")
    config_dict["dust3r_image_batch_size"] = image_batch_size
    config_dict["dust3r_overlap_percent"] = overlap_percent
    with open(os.path.join(annotation_dir, "config.json"), "w") as f:
        json.dump(config_dict, f)

if __name__ == '__main__':
    main()