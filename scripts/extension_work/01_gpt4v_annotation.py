import argparse
from orion.utils.misc_utils import get_all_annotation_folder_info
from orion.utils.openai_utils import (
    GPT4V_OKAMI,
    create_task_relevant_obj_prompt,
    sample_frames_from_video,
    create_tgt_obj_prompt,
    create_reference_obj_prompt,
    TESTING_MODEL,
    REASONING_MODEL,
    PROD_MODEL,
    encode_image,
    decode_image,
    create_messages_from_text_and_img,
    send_msg,
    ensure_json_parsability,
)


import cv2
import json
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_folder", default="aircraft_demo")
    parser.add_argument("--has_ref_obj", default=1)
    return parser.parse_args()


def get_obj_names_with_gpt(
    video_images,
    has_ref_obj,
    is_rgb=True,
    model_name=PROD_MODEL,
    max_tokens=1000,
    images_to_sample=5,
):
    # images, depths, masks, K, tracked_points = get_all_annotation_folder_info(
    #     annotation_folder
    # )
    vlm = GPT4V_OKAMI(model_name=model_name)
    sampled_images = sample_frames_from_video(video_images, sample_amount=images_to_sample)
    sampled_images_base64 = [
        vlm.preprocess_image(img, is_rgb=is_rgb, encoding="png")
        for img in sampled_images
    ]

    print("================ Task Relevant Object Names ================")
    text_prompt_list = [create_task_relevant_obj_prompt()]
    base64_img_list = sampled_images_base64
    role_list = ["user"]

    msgs = vlm.generate_msg(text_prompt_list, base64_img_list, role_list)
    vlm.check_msgs_mem(msgs)
    text_response, full_response_task_relevant_obj = vlm.generate_response(
        msgs, max_tokens=max_tokens
    )  # , max_tokens=100)
    text_response = ensure_json_parsability(text_response)
    print(f"Response: {text_response}")
    json_response = json.loads(text_response)
    objects_of_interest = json_response["objects_of_interest"]
    print("===============================================")


    print("================ Target Object Names ================")
    text_prompt_list = [create_tgt_obj_prompt(objects_of_interest)]
    role_list = ["user"]
    msgs = vlm.generate_msg(text_prompt_list, sampled_images_base64, role_list)
    vlm.check_msgs_mem(msgs)
    text_response, full_response_target_obj = vlm.generate_response(
        msgs, max_tokens=max_tokens
    )
    text_response = ensure_json_parsability(text_response)
    print(f"Response: {text_response}")
    json_response = json.loads(text_response)
    manipulate_object_name = json_response["manipulate_object_name"]

    if has_ref_obj:
        print("================ Reference Object Names ================")
        text_prompt_list = [
            create_reference_obj_prompt(manipulate_object_name, objects_of_interest)
        ]
        role_list = ["user"]
        msgs = vlm.generate_msg(text_prompt_list, sampled_images_base64, role_list)
        vlm.check_msgs_mem(msgs)
        text_response, full_response_ref_obj = vlm.generate_response(msgs, max_tokens=max_tokens)
        text_response = ensure_json_parsability(text_response)
        print(f"Response: {text_response}")
        json_response = json.loads(text_response)
        reference_object_name = json_response["reference_object_name"]
        print("===============================================")
    else:
        reference_object_name = None
        full_response_ref_obj = None
        
    full_responses = {}
    full_responses["task_relevant_obj"] = full_response_task_relevant_obj
    full_responses["target_obj"] = full_response_target_obj
    full_responses["reference_obj"] = full_response_ref_obj
    
    return manipulate_object_name, reference_object_name, full_responses