import argparse
import random
import torch
import os
import numpy as np
import traceback

from pathlib import Path
from PIL import Image
import init_path
from orion.utils.misc_utils import  get_first_frame_annotation, overlay_xmem_mask_on_image, resize_image_to_same_shape, plotly_draw_seg_image, sample_points_in_mask,get_annotation_info, get_depth_seq_from_human_demo
from orion.utils.o3d_utils import O3DPointCloud, load_reconstruction_info_from_human_demo, project3Dto2D, filter_pcd
import h5py

def sample_points_in_mask_wo_outlier(input_depth,
                                     input_annotation,
                                     camera_intrinsics,
                                     num_points,
                                     object_id=None):
    if object_id is None:
        segmented_depth = input_depth * (input_annotation != 0)
    else:
        segmented_depth = np.squeeze(input_depth) * (input_annotation == object_id)
    new_segmentation = filter_pcd(segmented_depth, camera_intrinsics) * object_id
    return sample_points_in_mask(new_segmentation, num_points, object_id)

def annotate_video(annotation_folder, num_track_points, save_video=True, use_dust3r=False, use_align3r=False):
    try:
        assert not (use_dust3r and use_align3r), "Cannot use both dust3r and align3r"
        first_frame, first_frame_annotation = get_first_frame_annotation(annotation_folder, use_dust3r=use_dust3r, use_align3r=use_align3r)

        info = get_annotation_info(annotation_folder)
        dataset_file_path = info["original_file"]

        first_depth = get_depth_seq_from_human_demo(
            dataset_file_path,
            use_dust3r=use_dust3r,
            use_align3r=use_align3r,
            annotation_folder=annotation_folder,
        )[0]

        if use_align3r:
            intrinsics_list = np.loadtxt(
                os.path.join(
                    annotation_folder,
                    "align3r",
                    os.path.basename(annotation_folder),
                    "pred_intrinsics.txt",
                )
            )
            intrinsics = intrinsics_list[0].reshape(3, 3)
        elif use_dust3r:
            demo_file = h5py.File(dataset_file_path, "r")
            intrinsics = demo_file["data/human_demo/obs/dust3r_intrinsics"][()][0]
        else:
            intrinsics = load_reconstruction_info_from_human_demo(dataset_file_path)["intrinsics"]

        points_list = []
        for object_id in range(1, first_frame_annotation.max() + 1):
            points = sample_points_in_mask_wo_outlier(first_depth, first_frame_annotation, intrinsics, num_track_points, object_id)
            points_list.append(points)

        points_list = np.concatenate(points_list, axis=0)
        points_name = os.path.join(annotation_folder, "points.pt")
        if use_align3r:
            points_name = os.path.join(annotation_folder, "align3r", "object_points.pt")
        torch.save(points_list, points_name)

        command = f"python scripts/cotracker_annotation.py --annotation-path {annotation_folder} --points-name {points_name}"
        if save_video:
            command += " --save-video"
        if use_dust3r:
            command += " --use-dust3r"
        if use_align3r:
            command += " --use-align3r"
        print(f"running: {command}")
        os.system(command)
    except Exception as e:
        print(f"Error in {annotation_folder}: {e}")
        print(traceback.format_exc())
        return False
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-folder", type=str, default=None)
    parser.add_argument("--num-track-points", type=int, default=40)
    parser.add_argument("--multiple-annotations", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--use-dust3r", action="store_true")
    parser.add_argument("--use-align3r", action="store_true")
    args = parser.parse_args()

    if args.no_video:
        save_video = False
    else:
        save_video = True
    error_folders = []

    if args.multiple_annotations:
        for annotation_folder in Path(args.annotation_folder).glob("*"):
            print(f"****Currently annotating {annotation_folder}*****")
            success = annotate_video(annotation_folder, args.num_track_points, save_video=save_video)
            if not success:
                error_folders.append(annotation_folder)
        print(f"Error folders: {error_folders}")
    else:
        annotate_video(
            args.annotation_folder,
            args.num_track_points,
            save_video=save_video,
            use_dust3r=args.use_dust3r,
            use_align3r=args.use_align3r,
        )

if __name__ == '__main__':
    main()
