import h5py
from PIL import Image
import argparse
import os
import json
import numpy as np
import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', required=True)
    parser.add_argument('--output-name', required=True)
    args = parser.parse_args()
    video_path = args.video_path
    output_name = args.output_name

    demo_file = h5py.File(os.path.join(os.path.dirname(video_path), output_name + ".hdf5"), "w")
    #convert video into numpy array of frames
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
    #create dataset for frames
    demo_file.create_dataset("data/human_demo/obs/agentview_rgb", data=frames)

if __name__ == "__main__":
    main()