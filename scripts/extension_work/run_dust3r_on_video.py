import sys
import os

import numpy as np
import os
import torch
import argparse

import cv2
import json
import math
import shutil
import uuid 
import h5py

def get_intrinsics(video_path=None, images_np=None, dust3r_path="/home/arisrei/research/RPL/ORION/third_party/dust3r", num_images=10, write_data_folder=None, annotation_folder=None, args=None):
    assert video_path is not None or images_np is not None, "Either video_path or images must be provided"
    assert video_path is None or images_np is None, "Only one of video_path or images must be provided"
    sys.path.append(dust3r_path)

    from dust3r.inference import inference, load_model
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from dust3r.utils.device import to_numpy

    if video_path is not None:
        vidcap = cv2.VideoCapture(video_path)
        success,image = vidcap.read()
        count = 0
        frames = []
        while success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #frames.append(image[:,:,::-1])
            frames.append(image[:,:,:])
            success,image = vidcap.read()
            count += 1
        frames = np.array(frames)
    else:
        frames = images_np


    original_resolution = frames.shape[1:3]
    #write frames to unique temp folder
    temp_folder = "/tmp/dust3r_temp" + str(uuid.uuid4())
    file_list = []
    picked_indices = []
    os.makedirs(temp_folder, exist_ok=True)
    for i in range(0, frames.shape[0], int(math.ceil(frames.shape[0]/num_images))):
        cv2.imwrite(os.path.join(temp_folder, f"{i}.png"), frames[0])
        file_list.append(os.path.join(temp_folder, f"{i}.png"))
        picked_indices.append(i)

    model_path = os.path.join(dust3r_path, "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
    device = 'cuda'
    
    batch_size = 1
    schedule = 'linear'
    lr = 0.01
    niter = 300

    model = load_model(model_path, device)
    images = load_images(file_list, size=512)

    #delete temp folder
    shutil.rmtree(temp_folder, ignore_errors=True)
    print("num images:", len(images))

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
    print(f"Original intrinsics: {intrinsics_curr[0]}")
    # extrinsics_curr = scene.get_im_poses().cpu().detach().numpy()

    # mask = to_numpy(scene.get_masks())
    # pts3d_curr = to_numpy(pts3d_curr)
    # imgs_curr = np.array(to_numpy(imgs_curr))
    depths_unormalized_curr = np.array(to_numpy(scene.get_depthmaps()))

    # pts3d_curr = np.concatenate([p[m] for p, m in zip(pts3d_curr, mask)])
    # col_curr = np.concatenate([p[m] for p, m in zip(imgs_curr, mask)])

    #scale the intrinsics to match the original resolution
    print(f"Original resolution: {original_resolution}")
    print(f"Current resolution: {imgs_curr[0].shape[:2]}")

    current_resolution = imgs_curr[0].shape[:2]
    x_scale = original_resolution[1] / current_resolution[1]
    y_scale = original_resolution[0] / current_resolution[0]
    intrinsics_curr[:,0,0] *= x_scale
    intrinsics_curr[:,1,1] *= y_scale
    intrinsics_curr[:,0,2] *= x_scale
    intrinsics_curr[:,1,2] *= y_scale

    print("Intrinsics adjusted to original resolution:", intrinsics_curr[0])
    if write_data_folder is not None:
        os.makedirs(write_data_folder, exist_ok=True)
        with open(os.path.join(write_data_folder, "dust3r_intrinsics.npy"), 'wb') as f:
            np.save(f, intrinsics_curr[0])
        #np.save(os.path.join(write_data_folder, "dust3r_intrinsics.npy"), intrinsics_curr[0])
        with open(os.path.join(write_data_folder, "dust3r_depths.npy"), 'wb') as f:
            np.save(f, depths_unormalized_curr)
        #np.save(os.path.join(write_data_folder, "dust3r_depths.npy"), depths_unormalized_curr)

        # save args
        with open(os.path.join(write_data_folder, "dust3r_args.json"), 'w') as f:
            json.dump(vars(args), f)

    if annotation_folder is not None:
        config = json.load(open(os.path.join(annotation_folder, "config.json")))
        original_file = h5py.File(config["original_file"], 'r+')

        if "data/human_demo/obs/dust3r_depth" in original_file:
            del original_file["data/human_demo/obs/dust3r_depth"]
        
        if "data/human_demo/obs/dust3r_rgb" in original_file:
            del original_file["data/human_demo/obs/dust3r_rgb"]

        original_file.create_dataset("data/human_demo/obs/dust3r_depth", data=depths_unormalized_curr)
        original_file.create_dataset("data/human_demo/obs/dust3r_rgb", data=imgs_curr)
        original_file.close()



    return [intrinsics_curr[0,0,0].item(), intrinsics_curr[0,1,1].item(), intrinsics_curr[0,0,2].item(), intrinsics_curr[0,1,2].item()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dust3r-path", type=str, default="/home/arisrei/research/RPL/ORION/third_party/dust3r")
    parser.add_argument("--cuda-device", type=int, default=7)
    parser.add_argument("--video-path", type=str)
    parser.add_argument("--write_data_folder", type=str, default=None)
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--annotation_folder", type=str, default=None)
    
    args = parser.parse_args()
    get_intrinsics(video_path=args.video_path, dust3r_path=args.dust3r_path, write_data_folder=args.write_data_folder, num_images=args.num_images, annotation_folder=args.annotation_folder, args=args)

if __name__ == '__main__':
    main()