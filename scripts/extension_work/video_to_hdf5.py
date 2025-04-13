import h5py
from PIL import Image
import argparse
import os
import json
import numpy as np
import cv2
from orion.utils.log_utils import get_orion_logger
from run_dust3r_on_video import get_intrinsics
from tqdm import tqdm
import shutil

ORION_LOGGER = get_orion_logger("orion")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', required=False)
    parser.add_argument('--output-name', required=False, default=None)
    parser.add_argument('--intrinsics', default=[1037.59562930012, 1026.70643336838, 636.632758971305, 380.040656309962], type=float, nargs=4, help='Intrinsics fx, fy, cx, cy')
    parser.add_argument('--video-list-path', default=None)
    parser.add_argument('--create-folder', default=True)
    parser.add_argument('--dust3r-intrinsics', default=False, action='store_true')
    args = parser.parse_args()
    assert args.video_path or args.video_list_path, "Either video_path or video_list_path must be provided"
    if args.video_list_path is not None:
        # txt file where each line contains path to video
        with open(args.video_list_path, "r") as f:
            video_list = f.readlines()
            video_list = [x.strip() for x in video_list]

    else:
        video_list = [args.video_path]    

    for video_path in tqdm(video_list):
        if args.output_name is None:
            output_name = os.path.basename(video_path).split(".")[0]
        else:
            output_name = args.output_name

        # get parent folder
        parent_folder = os.path.dirname(video_path)

        if args.create_folder:
            parent_folder = os.path.join(parent_folder, output_name)
            os.makedirs(parent_folder, exist_ok=True)

        demo_file = h5py.File(os.path.join(parent_folder, output_name + ".hdf5"), "w")

        # convert video into numpy array of frames
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        if image.shape[0] * image.shape[1] > 1920 * 1080:
            ORION_LOGGER.warning(
                "Video dimensions are too big. Downscaling"
            )
        count = 0
        frames = []
        while success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # if dimensions of the image are too big, downscale it but preserve aspect ratio
            if image.shape[0] * image.shape[1] > 1920 * 1080:
                scale_factor = (1920 * 1080) / (image.shape[0] * image.shape[1])
                image = cv2.resize(
                    image,
                    (
                        int(image.shape[1] * scale_factor),
                        int(image.shape[0] * scale_factor),
                    ),
                    interpolation=cv2.INTER_AREA,
                )

            frames.append(image[:, :, :])
            success, image = vidcap.read()
            count += 1

        frames = np.array(frames)

        # create dataset for frames
        demo_file.create_dataset("data/human_demo/obs/agentview_rgb", data=frames)

        if args.dust3r_intrinsics:
            # get intrinsics from dust3r
            intrinsics = get_intrinsics(images_np=frames)
            print("Intrinsics from dust3r: ", intrinsics)
            args.intrinsics = intrinsics

        # create json file for intrinsics as part of demo_file attrs
        if args.intrinsics is not None:
            data_config = {
                "intrinsics": {
                    "camera_rs_0": {
                        "fx": args.intrinsics[0],
                        "fy": args.intrinsics[1],
                        "cx": args.intrinsics[2],
                        "cy": args.intrinsics[3]
                    }
                }
            }

            demo_file['data'].attrs["data_config"] = json.dumps(data_config)

        demo_file.close()

        # move video file to parent folder
        if args.create_folder:
            shutil.copy(video_path, os.path.join(parent_folder, os.path.basename(video_path)))

if __name__ == "__main__":
    main()
