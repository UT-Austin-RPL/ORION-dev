import os
import argparse
import cv2
import numpy as np
import json
import h5py
from PIL import Image
import PIL
def dust3r_resize(img, size, square_ok=False):
    W1, H1 = img.size
    if size == 224:
        # resize short side to 224 (then crop)
        img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
    else:
        # resize long side to 512
        img = _resize_pil_image(img, size)
    W, H = img.size
    cx, cy = W//2, H//2
    if size == 224:
        half = min(cx, cy)
        img = img.crop((cx-half, cy-half, cx+half, cy+half))
    else:
        halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
        if not (square_ok) and W == H:
            halfh = 3*halfw/4
        img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
    return img

def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)



def create_video(frames, output_path, fps=30):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID', 'MJPG', etc.
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frames = frames[:,:,:,::-1]

    for frame in frames:
        out.write(frame)

    out.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-folder", type=str, default=None, required=True)
    parser.add_argument("--video-name", type=str, default="dust3r/video.mp4")
    parser.add_argument("--resize-masks", action="store_true")

    args = parser.parse_args()

    annotation_dir = args.annotation_folder
    config_dict = {}
    with open(os.path.join(annotation_dir, "config.json"), "r") as f:
        config_dict = json.load(f)
    demo_file = h5py.File(config_dict["original_file"], "r+")
    original_rgb = demo_file[f"data/human_demo/obs/agentview_rgb/"][()]
    resized_dims = np.array(dust3r_resize(Image.fromarray(original_rgb[0]), 512)).shape

    if args.resize_masks:
        masks_orig = np.load(os.path.join(annotation_dir, "masks.npz"))["arr_0"]
        resized_masks = np.zeros((original_rgb.shape[0], resized_dims[0], resized_dims[1]), dtype=np.uint8)
        for i in range(original_rgb.shape[0]):
            resized_masks[i] = np.array(Image.fromarray(masks_orig[i]).resize((resized_dims[1], resized_dims[0]), resample=PIL.Image.NEAREST))
        
        if "dust3r_mask" in demo_file["data/human_demo/obs/"]:
            del demo_file["data/human_demo/obs/dust3r_mask"]
        demo_file.create_dataset("data/human_demo/obs/dust3r_mask", data=resized_masks)

        

        print(masks_orig.shape, resized_masks.shape)

    else:
        dust3r_rgb = np.zeros((original_rgb.shape[0], resized_dims[0], resized_dims[1], 3), dtype=np.uint8)
        for i in range(original_rgb.shape[0]):
            dust3r_rgb[i] = np.array(dust3r_resize(Image.fromarray(original_rgb[i]), 512))
        print(original_rgb.shape, dust3r_rgb.shape)
        create_video(dust3r_rgb, os.path.join(annotation_dir, args.video_name))

        config_dict["video_file_dust3r"] = os.path.join(annotation_dir, args.video_name)
        with open(os.path.join(annotation_dir, "config.json"), "w") as f:
            json.dump(config_dict, f)
        print("Saving to", os.path.join(annotation_dir, args.video_name))
    demo_file.close()

if __name__ == '__main__':
    main()

