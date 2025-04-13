import h5py
from PIL import Image
import argparse
import os
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-folder', required=True)
    args = parser.parse_args()
    
    annotations_folder = args.annotation_folder

    config_dict = ""
    assert(os.path.exists(annotations_folder)), f"Annotation folder {annotations_folder} does not exist."
    with open(os.path.join(annotations_folder, "config.json"), "r") as f:
        config_dict = json.load(f)
    
    demo_file = h5py.File(config_dict["original_file"], "r")

    output_folder = os.path.join(os.path.dirname(config_dict["original_file"]), "images")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video_np = demo_file.get("data/human_demo/obs/agentview_rgb")[()]
    for i in range(video_np.shape[0]):
        img = Image.fromarray(video_np[i])
        img.save(os.path.join(output_folder, f"image{i}.png"))
    print("saved in ", output_folder)

    with open(os.path.join(annotations_folder, "config.json"), "w") as f:
        config_dict['images_folder'] = output_folder
        json.dump(config_dict, f)
    demo_file.close()

if __name__ == "__main__":
    main()