'''
    Helpful tutorials: https://cookbook.openai.com/
'''
from datetime import datetime
import json
import os
import sys
import base64
import time
import cv2
import numpy as np
from typing import List, Optional, Union
from termcolor import colored
from easydict import EasyDict
from openai import OpenAI
from orion import ROOT_PATH
from PIL import Image

TESTING_MODEL = "gpt-4o-mini" # to save api credits
PROD_MODEL = "gpt-4o"
REASONING_MODEL = 'o1'

models = [TESTING_MODEL, REASONING_MODEL, PROD_MODEL]

class GPT4V_OKAMI:
    def __init__(
        self,
        model_name: str = TESTING_MODEL
    ):
        super().__init__()
        # only tested with gpt-4o but most of the boilerplate code should work with other models as well.
        assert model_name in models, f"Model {model_name} not supported."
        self.model_name = model_name
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))
        self.begin_new_dialog()
    
    def begin_new_dialog(self):
        self.history_msgs = [{
            "role": "system",
            "content": "You are a helpful assistant."
        }]

    @property
    def available_encodings(self):
        # additional available encodings are: 'gif', 'webp'
        return ['jpg', 'png', 'jpeg']

    def verify_msg(self, msgs: List[dict]):
        for msg in msgs:
            assert msg['role'] in ['user', 'system', 'assistant'], f"role {msg['role']} not recognized."
            for content in msg['content']:
                # GPT4V only supports text for system and assistant at the moment
                if (msg['role'] == 'system') or (msg['role'] == 'assistant'):
                    assert isinstance(content, str), f"content {content} not recognized for role {msg['role']}."
                elif msg['role'] == 'user':
                    assert content['type'] in ['text', 'image_url'], f"type {content['type']} not recognized."
                    if content['type'] == 'image_url':
                        assert isinstance(content['image_url'], dict), f"image_url {type(content['image_url'])} not recognized."
                    elif content['type'] == 'text':
                        assert isinstance(content['text'], str), f"text {content['text']} not recognized."
                    else:
                        raise ValueError(f"type {content['type']} not recognized.")
        return True

    def check_msgs_mem(self, msgs: list):
        # calculate the memory of the msgs
        total_mem = sys.getsizeof(msgs) / 1024 / 1024
        assert total_mem < 20, "total msg should be less than 20Mb"
        return total_mem

    def preprocess_image(self, img: np.ndarray, is_rgb: bool, encoding: str = 'png'):
        '''
            Convert the np.ndarray image to base64 string.
        '''
        assert encoding in self.available_encodings, f"Encoding {encoding} not supported."
        if is_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.'+encoding, img)
        return base64.b64encode(buffer).decode('utf-8')

    def print_prompts(self, text_prompt_list: List[str]):
        print("========== Text Prompts ==========")
        for i, text in enumerate(text_prompt_list):
            print(f"{i+1}. {text}")
        print("==================================")
        

    def generate_msg(
            self,
            text_prompt_list: List[str],
            base64_img_list: Union[List[str], List[List[str]]],
            role_list: List[str] = None,
            encoding_list: List[str] = None,
            detail_list: List[str] = None,
            all_images_to_prompt: bool = True
        ):
        '''
            Given a list of text prompts and base64 images, wrap them in the required format.
            - We assume that each text is corresponding to single image. This can be tweaked by adding more content with type 'image_url' in the msg.
            - You can also specify the role of the user, system, or assistant. For system and assistant, only text is supported.
        '''
        self.print_prompts(text_prompt_list)
        if role_list is None:
            role_list = ["user"] * len(text_prompt_list)
        if encoding_list is None:
            encoding_list = ["jpg"] * len(base64_img_list)
        if detail_list is None: # detail_list can be low or high.
            detail_list = ["high"] * len(text_prompt_list)
        
        if "low" in detail_list:
            timestamp = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            temp_folder = os.path.join(ROOT_PATH, "junk", timestamp)
            os.makedirs(temp_folder, exist_ok=True)
            print(f"Saving low resolution images to {temp_folder}")
            
        for i, detail in enumerate(detail_list):
            if detail == "low":
                img = cv2.imdecode(np.frombuffer(base64.b64decode(base64_img_list[i]), np.uint8), cv2.IMREAD_COLOR)
                img = cv2.resize(img, (512, 512))
                temp_file_path = os.path.join(temp_folder, f"temp_image_{i}.jpg")
                cv2.imwrite(temp_file_path, img)
            
        
        msgs = self.history_msgs.copy()
        for i in range(len(text_prompt_list)):
            msg = {
                "role": role_list[i],
                "content": [
                    {
                        "type": "text",
                        "text": text_prompt_list[i]
                    }
                ]
            }
            if (role_list[i] == "user") and (base64_img_list[i] is not None):
                if isinstance(base64_img_list[i], list):
                    for img in base64_img_list[i]:
                        msg['content'].append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{encoding_list[i]};base64,{img}",
                                "detail": detail_list[i]
                            }
                        })
                else:
                    if all_images_to_prompt:
                        for img in base64_img_list:
                            msg['content'].append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{encoding_list[i]};base64,{img}",
                                    "detail": detail_list[i]
                                }
                            })
                    else:    
                        msg['content'].append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{encoding_list[i]};base64,{base64_img_list[i]}",
                                "detail": detail_list[i]
                            }
                        })

            msgs.append(msg)

        self.check_msgs_mem(msgs)
        return msgs

    def generate_response(
            self,
            msgs: List[dict],
            max_tokens: Optional[int] = None,
            temperature: float = 0,
            fake_response: bool = False,
        ):
        assert self.verify_msg(msgs), "msgs not verified"
        if fake_response:
            # create a fake response
            fake_response = EasyDict({
                "choices": [
                    {
                        "finish_reason": "fake",
                        "message": {
                            "content": "fake"
                        }
                    }
                ]
            })
            return fake_response
        try:
            print("Generating response...")
            
            if self.model_name == REASONING_MODEL:
                if max_tokens is not None:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=msgs,
                        max_completion_tokens=max_tokens,
                        # response_format={"type": "json_object"}, # Not supported in GPT-4 Vision
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=msgs,
                        # response_format={"type": "json_object"}, # Not supported in GPT-4 Vision
                    )
            else:
                if max_tokens is not None:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=msgs,
                        max_tokens=max_tokens,
                        # response_format={"type": "json_object"}, # Not supported in GPT-4 Vision
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=msgs,
                        # response_format={"type": "json_object"}, # Not supported in GPT-4 Vision
                    )
        except Exception as e:
            print(colored(f"[ERROR] {e}", 'red'))

        if response.choices[0].finish_reason == 'length':
            print(colored(f"[WARNING] Output may be incomplete due to token limit. Consider increasing your max_tokens", 'red'))
        elif response.choices[0].finish_reason == 'fake':
            print(colored(f"[WARNING] Fake response detected.", 'red'))
        elif response.choices[0].finish_reason == 'content_filter':
            print(colored(f"[WARNING] content was omitted due to a flag from openai content filters", 'red'))
        elif response.choices[0].finish_reason == 'stop':
            pass
        
        self.history_msgs = msgs.copy() # add to history
        self.history_msgs.append({ # add the response to history
            "role": "assistant",
            "content": response.choices[0].message.content
        })
        return self.cleanup_response(response.choices[0].message.content), response

    def cleanup_response(self, text_response: str):
        '''
            Given the response string, cleanup the response and return the cleaned response.
            Custom cleanup can be done here, e.g., json parsing.
        '''
        # some clean up
        return text_response
    
    def run(self, 
            text_prompt_list: List[str],
            base64_img_list: Union[List[str], List[List[str]]],
            max_tokens: int = 500,
            role_list: List[str] = None,
            encoding_list: List[str] = None,
            detail_list: List[str] = None,
        ):
        """
        high-level API: given a list of text prompts and base64 images, generate a response.
        """
        msgs = self.generate_msg(text_prompt_list, base64_img_list, role_list, encoding_list, detail_list)
        text_response, full_response = self.generate_response(msgs, max_tokens=max_tokens)
        return text_response

def create_messages_from_text_and_img(text_list: List[str], img_list: List[np.ndarray], is_rgb: bool = True):
    messages = []
    for i in range(len(text_list)):
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text_list[i]
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_list[i], type='np')}"},
                }
            ]
        })
    return messages


def ensure_json_parsability(text_response):
    return text_response.replace("```json", "").replace("```","")
    

def send_msg(messages, model=TESTING_MODEL):
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response
    
def encode_image(image_path_or_np, type='np', is_rgb=True):
    if type == 'np':
        if is_rgb:
            image_path_or_np = cv2.cvtColor(image_path_or_np, cv2.COLOR_RGB2BGR)
        return base64.b64encode(cv2.imencode('.jpg', image_path_or_np)[1]).decode("utf-8")    
    else:
        with open(image_path_or_np, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

def decode_image(base64_str: str, is_rgb: bool = True) -> np.ndarray:
    """
    Decode a base64 string to an np.ndarray image.
    """
    img_data = base64.b64decode(base64_str)
    img_array = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if is_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
        

def sample_frames_from_video(frames: np.ndarray, sample_amount: int = 10):
    indices = np.linspace(0, len(frames) - 1, sample_amount, dtype=int)
    print(f"Sampling frames at indices: {indices}")
    chosen_frames = frames[indices].copy()
    chosen_frames_with_text = []
    
    # Add black bar with frame number text to each frame
    for i, idx in enumerate(indices):
        h, w = chosen_frames[i].shape[:2]
        bar_height = 40  # Height of the black bar
        
        # Create new image with space for the black bar
        new_frame = np.zeros((h + bar_height, w, 3), dtype=np.uint8)
        
        # Copy original frame to the top portion
        new_frame[:h, :] = chosen_frames[i]
        
        # Add black bar at the bottom (already black from np.zeros)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Frame {idx}"
        font_scale = 0.8
        text_thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
        
        # Center text horizontally, position vertically in the bar
        text_x = (w - text_size[0]) // 2
        text_y = h + (bar_height + text_size[1]) // 2
        
        cv2.putText(new_frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), 
                    text_thickness, cv2.LINE_AA)
        
        # Replace the frame with the new one
        chosen_frames_with_text.append(new_frame)
    chosen_frames_with_text = np.array(chosen_frames_with_text)
    
    return chosen_frames_with_text
    

def formatting_prompt():
    return """
There is no need to make your output parseable by latex or markdown. So no need for "```json" or "```" at the beginning and end of your response."""

def create_task_relevant_obj_prompt():
    resp = """
You need to analyze what the human is doing in the images, then tell me: 1. All the objects in front of the scene (mostly on a plane similar to a table). You should ignore the background objects. 2. The objects of interest. They should be a subset of your answer to the first question. They are likely the objects manipulated by the human or objects that interact with other objects. Objects that the human uses (like utensils) to directly manipulate other objects should NOT be included as task relevant objects. Note that there are irrelevant objects in the scene, such as objects that are not interacted with by the human and/or the other objects. You should ignore the irelevant objects. Make sure to include a color and texture description of the relevant objects. This is especially important if there are many instances of a single type of object. For example, if there are blue, green, and yellow balls, be sure to specify each one individually. Do NOT just say "colorful balls". Your output format is:

{
    "description": "The human is xxx.",
    "all_objects": ["OBJECT1", "OBJECT2", ...], 
    "objects_of_interest": ["OBJECT1", "OBJECT3", ...],
}

Ensure the response can be parsed by Python 'json.loads', e.g.: no trailing commas, no single quotes, etc.

Example:
{
    "description": "The human is picking up a red ball with blue dots and putting it in a green cup",
    "all_objects": ["blue ball with stripes", "small green ball", "red ball with blue dots", "green cup"], 
    "objects_of_interest": ["red ball with blue dots", "green cup"],
}
"""
    
    resp += formatting_prompt()
    return resp

def create_tgt_obj_prompt(task_relevant_object_candidates: List[str]):
    resp = f"""
The following images shows a manipulation motion, where the human is manipulating an object or objects. 

Your task is to determine which objects are being manipulated in the images below. You need to choose from the following objects: {task_relevant_object_candidates}.

Tips: A manipulated object is an object that the human is interacting with, such as picking up, moving, or pressing, and it is in contact with the human or an object (like a utensil) that the human is using. If there are multiple objects of the same type, be sure to specify the color, texture, etc. that correspond to the manipulated object

Your output format is:
{{
"manipulate_object_names": ["MANIPULATE_OBJECT_DESCRIPTIVE_NAME_1", "MANIPULATE_OBJECT_DESCRIPTIVE_NAME_2", ...],
}}

Example:
{{
    "manipulate_object_names": ["red ball with blue dots", "green cup"],
}}

Ensure the response can be parsed by Python 'json.loads', e.g.: no trailing commas, no single quotes, etc.
    """
    
    resp += formatting_prompt()
    return resp

def create_reference_obj_prompt(manipulate_obj_names: List[str], task_relevant_object_candidates: List[str]):
    resp = f"""
The following images shows a manipulation motion, where the human is manipulating the objects in the list - {manipulate_obj_names}.

Please identify the reference object in the image below, which could be an object on which to place an object in the list {manipulate_obj_names}, or an object that something in the list {manipulate_obj_names} is interacting with. You need to first identify whether there is a reference object. If so, you need to output the reference object's name chosen from the following objects: {task_relevant_object_candidates}.
Note that there may not necessarily be a reference object. Examples of reference objects are: a coaster, if the motion involves picking up a cup and placing it on a coaster; a bowl, if the motion involves picking up a bottle and pouring it into the bowl.

Your output format is:
{{
"reference_object_names": ["REFERENCE_OBJECT_NAME_1", "REFERENCE_OBJECT_NAME_2", ...] or ["None"],
"reasoning": "Here, you will explain your reasoning for choosing these objects as the reference object, or for saying that there are no reference objects"
}}

Example:
{{
    "reference_object_names": ["white circular plate", "brown coaster"],
    "reasoning": The white circular plate is a reference object because the human is placing the red ball on ti. The brown coaster is a reference object because the human is placing the green cup on it.
}}

Ensure the response can be parsed by Python 'json.loads', e.g.: no trailing commas, no single
quotes, etc.
"""
    
    resp += formatting_prompt()
    return resp

def get_obj_names_with_gpt(
    video_images,
    may_have_ref_obj=True,
    is_rgb=True,
    model_name=PROD_MODEL,
    max_tokens=1000,
    images_to_sample=5,
    debug=False,
    debug_save_folder=None,
):
    # images, depths, masks, K, tracked_points = get_all_annotation_folder_info(
    #     annotation_folder
    # )
    vlm = GPT4V_OKAMI(model_name=model_name)
    sampled_images = sample_frames_from_video(video_images, sample_amount=images_to_sample)
    if debug:
        for i, img in enumerate(sampled_images):
            Image.fromarray(sampled_images[i]).save(os.path.join(debug_save_folder, f"sampled_image_{i}.jpg"))
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
    manipulate_object_names = json_response["manipulate_object_names"]

    if may_have_ref_obj:
        print("================ Reference Object Names ================")
        text_prompt_list = [
            create_reference_obj_prompt(manipulate_object_names, objects_of_interest)
        ]
        role_list = ["user"]
        msgs = vlm.generate_msg(text_prompt_list, sampled_images_base64, role_list)
        vlm.check_msgs_mem(msgs)
        text_response, full_response_ref_obj = vlm.generate_response(msgs, max_tokens=max_tokens)
        text_response = ensure_json_parsability(text_response)
        print(f"Response: {text_response}")
        json_response = json.loads(text_response)
        reference_object_names = json_response["reference_object_names"] if json_response["reference_object_names"] != "None" else None
        print("===============================================")
    else:
        reference_object_names = [None]
        full_response_ref_obj = None
        
    full_responses = {}
    full_responses["task_relevant_obj"] = str(full_response_task_relevant_obj)
    full_responses["target_obj"] = str(full_response_target_obj)
    full_responses["reference_obj"] = str(full_response_ref_obj)
    
    return manipulate_object_names, reference_object_names, full_responses