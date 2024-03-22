import h5py
from PIL import Image
import argparse
import os
import json
import pathlib
import pycolmap

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
    dataset_folder = os.path.dirname(config_dict["original_file"])
    image_folder = os.path.join(dataset_folder, "images")
    if not os.path.exists(image_folder):
        print("image_folder does not exist")
        return
    
    output_path = pathlib.Path(os.path.join(dataset_folder, "colmap"))
    image_dir = pathlib.Path(image_folder)

    output_path.mkdir()
    mvs_path = output_path / "mvs"
    database_path = output_path / "database.db"

    pycolmap.extract_features(database_path, image_dir)
    pycolmap.match_exhaustive(database_path)
    maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
    maps[0].write(output_path)
    # dense reconstruction
    pycolmap.undistort_images(mvs_path, output_path, image_dir)
    pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
    pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)
    

    




if __name__ == "__main__":
    main()