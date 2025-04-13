import argparse
import os
import numpy as np
import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', required=False)
    parser.add_argument('--output-folder', required=False, default=None)
    args = parser.parse_args()
    assert args.video_path or args.video_list_path, "Either video_path or video_list_path must be provided"
    
    os.makedirs(args.output_folder, exist_ok=True)
    video_path = args.video_path

    #convert video into numpy array of frames
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    frames = []
    while success:
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #frames.append(image[:,:,::-1])
        frames.append(image[:,:,:])
        success,image = vidcap.read()
        count += 1
    frames = np.array(frames)

    # write to output folder
    # make output names in ascending alphabetical order of frame number

    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(args.output_folder, f"{i:04d}.png"), frame)


if __name__ == "__main__":
    main()