import torch
import cv2
from PIL import Image
import argparse
import json
import os
import h5py
import numpy as np
import sys
import matplotlib
import shutil

import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), "../third_party/ZoeDepth"))

from zoedepth.utils.misc import save_raw_16bit

from torchvision.transforms import ToTensor

def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations-folder', required=True)

    args = parser.parse_args()

    annotations_folder = args.annotations_folder

    config_dict = ""
    assert(os.path.exists(annotations_folder)), f"Annotation folder {annotations_folder} does not exist."
    with open(os.path.join(annotations_folder, "config.json"), "r") as f:
        config_dict = json.load(f)
    
    demo_file = h5py.File(config_dict["original_file"], "r+")
    video_np = demo_file.get("data/human_demo/obs/agentview_rgb")[()]
    video_np = video_np[..., ::-1] #RGB to BGR
    
    if "data/human_demo/obs/zoe_depth" in demo_file:
        del demo_file["data/human_demo/obs/zoe_depth"]
    if "data/human_demo/obs/rescale_max_min" in demo_file:
        del demo_file["data/human_demo/obs/rescale_max_min"]

    if os.path.isdir(os.path.join(args.annotations_folder,"zoe_output")):
        shutil.rmtree(os.path.join(args.annotations_folder,"zoe_output"))
    os.mkdir(os.path.join(args.annotations_folder,"zoe_output"))


    rescale_max_min = np.zeros((video_np.shape[0], 2), dtype='float32')
    depth_np = np.zeros((video_np.shape[0:3] + tuple([1])), dtype='uint16')
    depth_rgb = np.zeros((video_np.shape[0:3] + tuple([3])), dtype='uint8')

    #print(depth_np.shape)

    model_zoe_n = torch.hub.load("./third_party/ZoeDepth", "ZoeD_N", source="local", pretrained=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_n.to(DEVICE)

    output_folder = os.path.join(annotations_folder, "zoe_output")
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
    os.mkdir(output_folder)
    for i in range(video_np.shape[0]):
        curr_img = Image.fromarray(video_np[i])

        orig_size = curr_img.size
        X = ToTensor()(curr_img)
        X = X.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = zoe.infer(X).cpu().numpy()
            #out = zoe.infer(X)

        #print(np.unique(out), out.shape)

        out_uint16 = ((out-np.min(out))/(np.max(out)-np.min(out)))
        out_uint16 = (out_uint16 * 65535.0).astype(np.uint16)
        #print(np.unique(out_uint16))
        
        old_max = np.max(out)
        old_min = np.min(out)
        rescale_max_min[i,:] = [old_max, old_min]


        
        # rescale_back = ((out_uint16 / 65535.0) * (old_max-old_min)) + old_min
        # print("rescale_back", np.unique(rescale_back), rescale_back.shape)

        #dif = np.abs(rescale_back - out) / out
        #print("dif", dif.shape, np.sum(dif)/dif.size)



        depth_np[i, :, :, :] = np.transpose(out_uint16[0, :, :, :], (1,2,0))

        
        #depth_pil = zoe.infer_pil(curr_img, output_type="pil")  # as 16-bit PIL Image
        
        #depth_pil = np.array(depth_pil)
        #depth_pil = colorize(depth_pil)
        #print(depth_pil.shape, "depth_pil")
        #depth_rgb[i] = depth_pil[:,:,0:3]

        # depth_pil = Image.fromarray(depth_pil)

        # depth_pil.save(os.path.join(output_folder, f'image_{i}.png'))

    demo_file.create_dataset("data/human_demo/obs/zoe_depth", data=depth_np)
    demo_file.create_dataset("data/human_demo/obs/rescale_max_min", data=rescale_max_min)
    demo_file.close()

    

    #np.save(os.path.join(output_folder, "depth_zoe.npy"), depth_np)
    # with open(os.path.join(annotations_folder, "config.json"), 'w') as f:
    #     json.dump(config_dict, f)


    # num_frames, height, width, _ = depth_rgb.shape
    # filename1 = os.path.join(output_folder, "zoe_depth.mp4")
    # codec_id = "mp4v" # ID for a video codec.
    # fourcc = cv2.VideoWriter_fourcc(*codec_id)
    # out = cv2.VideoWriter(filename1, fourcc=fourcc, fps=20, frameSize=(width, height))

    # for i in range(num_frames):
    #     out.write(depth_rgb[i])
    # out.release()

if __name__ == "__main__":
    main()