"""
A simple user interface for XMem
"""

import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2
import json

import numpy as np
import argparse
import shutil
import torch

from PIL import Image
import matplotlib.pyplot as plt
from easydict import EasyDict
import init_path
from orion.utils.misc_utils import (
    get_palette,
    load_first_frame_from_hdf5_dataset,
    export_video_from_hdf5_dataset,
    load_first_frame_from_human_hdf5_dataset,
    export_video_from_human_hdf5_dataset,
    load_image_from_human_hdf5_dataset,
    load_video_from_hdf5_dataset,
    load_video_from_human_hdf5_dataset,
    overlay_xmem_mask_on_image,
)
from orion.algos.grounded_sam_wrapper import GroundedSamWrapper
import argparse
import uuid
from orion.utils.openai_utils import GPT4V_OKAMI, create_task_relevant_obj_prompt, create_tgt_obj_prompt, create_reference_obj_prompt, get_obj_names_with_gpt

torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser(description="A simple user interface for XMem")
    parser.add_argument("--annotator", choices=["GSAM", "VLM"], help="Select the method of annotation")
    parser.add_argument("--image", help="Path to an image file")
    parser.add_argument("--video", help="Path to a video file")
    parser.add_argument("--demo", help="Path to a demo file")
    parser.add_argument("--rollout", help="Path to a rollout file")
    parser.add_argument(
        "--human_demo",
        default="datasets/iphones/long_horizon_boat/iphone_long_horizon_boat_18_demo.hdf5",
        help="Path to a human demo file",
    )
    parser.add_argument("--output_folder_name", help="If provided, save the annotation to this folder. Otherwise, save to a folder with same name as demonstration file", default=None)
    
    # GSAM Args
    parser.add_argument("--text", nargs="+", help="Describe what to segment")
    parser.add_argument("--tgt_obj_text", nargs="+", help="Describe the target object text. Cannot be used with --text")
    parser.add_argument("--ref_obj_text", nargs="+", help="Describe the reference object text. Cannot be used with --text")
    
    # VLM Args
    parser.add_argument("--images_to_sample", default=10, type=int, help="Number of images to sample from the video for input to VLM")
    
    parser.add_argument(
        "--bgr", default=True, action="store_true", help="Use BGR color space"
    )
    parser.add_argument("--no-bgr", dest="bgr", action="store_false")
    parser.add_argument(
        "--combine-all",
        default=False,
        action="store_true",
        help="Combine all generated masks",
    )
    parser.add_argument(
        "--text-threshold", default=0.4, type=float, help="Text threshold"
    )
    parser.add_argument(
        "--box-threshold", default=0.51, type=float, help="Box threshold"
    )
    parser.add_argument("--filter-threshold", default=200, type=int)
    parser.add_argument("--debug", default=0, type=int)

    return parser.parse_args()


def main():
    args = parse_args()

    assert (args.text is None) or (args.tgt_obj_text is None and args.ref_obj_text is None), "Cannot use --text with --tgt_obj_text or --ref_obj_text"

    mode = ""
    if args.image is not None:
        mode = "image"
    elif args.video is not None:
        mode = "video"
    elif args.demo is not None:
        mode = "demo"
    elif args.human_demo is not None:
        mode = "human_demo"
    elif args.rollout is not None:
        mode = "rollout"

    annotation_folder = f"annotations/{mode}"
    tmp_folder = "tmp_images" + str(uuid.uuid4())

    if args.output_folder_name:
        annotation_path = os.path.join(
            annotation_folder, args.output_folder_name
        )
    elif mode == "image":
        annotation_path = os.path.join(
            annotation_folder, args.image.split("/")[-1].split(".")[0]
        )
    elif mode == "video":
        annotation_path = os.path.join(
            annotation_folder, args.video.split("/")[-1].split(".")[0]
        )
    elif mode == "demo":
        annotation_path = os.path.join(
            annotation_folder, args.demo.split("/")[-1].split(".")[0]
        )
    elif mode == "human_demo":
        annotation_path = os.path.join(
            annotation_folder, args.human_demo.split("/")[-1].split(".")[0]
        )
    elif mode == "rollout":
        annotation_path = os.path.join(
            annotation_folder, args.rollout.split("/")[-1].split(".")[0]
        )

    tmp_path = tmp_folder

    os.makedirs(annotation_path, exist_ok=True)
    os.makedirs(tmp_path, exist_ok=True)
    os.makedirs(os.path.join(tmp_path, "images"), exist_ok=True)

    if mode == "image":
        first_frame = cv2.imread(args.image)
    elif mode == "rollout":
        first_frame = cv2.imread(args.rollout)
    elif mode == "video":
        _, first_frame = cv2.VideoCapture(args.video).read()
    elif mode == "demo":
        first_frame = load_first_frame_from_hdf5_dataset(
            args.demo, demo_idx=args.demo_idx, bgr=args.bgr
        )
    elif mode == "human_demo":
        first_frame = load_first_frame_from_human_hdf5_dataset(
            args.human_demo, bgr=args.bgr
        )
        last_frame = load_image_from_human_hdf5_dataset(dataset_name=args.human_demo, frame_idx=-1, image_name="agentview_rgb", bgr=args.bgr)
        Image.fromarray(last_frame).save(
            os.path.join(os.path.join(tmp_path, "images", "last_frame.jpg"))
        )

    print("Original image is of shape", first_frame.shape)
    Image.fromarray(first_frame).save(
        os.path.join(os.path.join(tmp_path, "images", "frame.jpg"))
    )

    args.images = tmp_path
    args.workspace = tmp_path

    print(args.images, tmp_folder)

    # launch_gui(args)
    print("Annotating the image with text input: ", args.text)
    wrapper = GroundedSamWrapper()
    if args.annotator == "GSAM":
        if args.text:
            final_mask_image = wrapper.segment(
                first_frame,
                args.text,
                concat_masks=args.combine_all,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                filter_threshold=args.filter_threshold,
            )
        elif args.tgt_obj_text and args.ref_obj_text:
            final_mask_image_tgt: Image.Image = wrapper.segment(
                first_frame,
                args.tgt_obj_text,
                concat_masks=True,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                filter_threshold=args.filter_threshold,
            )
            final_mask_image_ref: Image.Image = wrapper.segment(
                first_frame,
                args.ref_obj_text,
                concat_masks=True,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                filter_threshold=args.filter_threshold,
            )

            # combine the masks together
            mask_tgt = np.array(final_mask_image_tgt)
            mask_ref = np.array(final_mask_image_ref)
            last_tgt_color = mask_tgt.max()
            mask_ref[mask_ref > 0] += last_tgt_color

            final_mask_image = Image.fromarray(mask_tgt + mask_ref)
            final_mask_image.putpalette(get_palette())
        else:
            raise ValueError("No text input provided")
    elif args.annotator == "VLM":
        images = load_video_from_human_hdf5_dataset(
            dataset_name=args.human_demo,
            demo_idx=0,
            image_name="agentview_rgb",
            bgr=args.bgr
        )
        images_for_vlm_debug_folder = os.path.join(
            annotation_path, "images_for_vlm_debug"
        )
        
        if args.debug:
            os.makedirs(images_for_vlm_debug_folder, exist_ok=True)
            
        manipulate_object_names, reference_object_names, full_responses = (
            get_obj_names_with_gpt(
                video_images=images,
                is_rgb=not args.bgr,
                images_to_sample=args.images_to_sample,
                debug=args.debug,
                debug_save_folder=images_for_vlm_debug_folder,
            )
        )
        object_names_text = []
        for manipulate_object_name in manipulate_object_names:
            object_names_text.append(manipulate_object_name)
        for reference_object_name in reference_object_names:
            if reference_object_name is not None:
                object_names_text.append(reference_object_name)

        object_names_text = [
            name for name in object_names_text if name is not None
        ]  # remove None values

        final_mask_image = wrapper.segment(
            first_frame,
            prompts=object_names_text,
            concat_masks=args.combine_all,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            filter_threshold=args.filter_threshold,
        )

        args.text = object_names_text

    # print("Mask image is of size", final_mask_image.shape)
    os.makedirs(os.path.join(tmp_path, "masks"), exist_ok=True)
    final_mask_image.save(os.path.join(tmp_path, "masks", "frame.png"))
    overlay_image = overlay_xmem_mask_on_image(
        first_frame, np.array(final_mask_image), use_white_bg=True, rgb_alpha=0.3
    )

    try:
        plt.imshow(overlay_image)
        plt.show()
    except:
        pass

    # copy a image from a folder to another
    shutil.copyfile(
        os.path.join(tmp_path, "images", "frame.jpg"),
        os.path.join(annotation_path, "frame.jpg"),
    )
    shutil.copyfile(
        os.path.join(tmp_path, "masks", "frame.png"),
        os.path.join(annotation_path, "frame_annotation.png"),
    )
    shutil.copyfile(
        os.path.join(tmp_path, "images", "last_frame.jpg"),
        os.path.join(annotation_path, "last_frame.jpg")
    )

    print("Annotation saved to ", os.path.join(annotation_path, "frame_annotation.png"))
    with open(os.path.join(annotation_path, "config.json"), "w") as f:
        config_dict = {"mode": mode}
        if mode == "image":
            config_dict["original_file"] = args.image
        elif mode == "rollout":
            config_dict["original_file"] = args.rollout
        elif mode == "video":
            config_dict["original_file"] = args.video
            config_dict["video_file"] = args.video
        elif mode == "demo":
            config_dict["original_file"] = args.demo
            video_path = export_video_from_hdf5_dataset(
                dataset_name=args.demo,
                demo_idx=args.demo_idx,
                video_path=annotation_path,
                video_name=args.demo.split("/")[-1].split(".")[0],
                bgr=args.bgr,
            )
            config_dict["video_file"] = video_path
        elif mode == "human_demo":
            config_dict["original_file"] = os.path.abspath(args.human_demo)
            video_path = export_video_from_human_hdf5_dataset(
                dataset_name=args.human_demo,
                video_path=annotation_path,
                video_name=args.human_demo.split("/")[-1].split(".")[0],
                bgr=args.bgr,
            )

            config_dict["video_file"] = video_path

        if args.annotator == "VLM":
            config_dict['vlm_responses'] = full_responses
        config_dict["text"] = args.text

        json.dump(config_dict, f)
    # remove the folder
    shutil.rmtree(tmp_path)


if __name__ == "__main__":
    main()
