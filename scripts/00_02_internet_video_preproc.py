import argparse
import subprocess
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from extension_work.create_foundationpose_formatting import create_foundationpose_formatting
from extension_work.create_transparent_bg_from_mask import create_transparent_bg_from_mask_annotation_folder
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Path to a video file")
    parser.add_argument("--skip_ran_steps", help="Skip the steps that have already been ran", action="store_true", default=False)
    parser.add_argument("--skip_specific_steps", default=[], type=float, nargs="+")
    parser.add_argument("--output_folder_name", help="If provided, save the annotation to this folder. Otherwise, save to a folder with same name as video", default=None)
    parser.add_argument("--debug", default=0, type=int)


    # GSAM Annotation Args
    parser.add_argument(
        "--text-threshold", default=0.3, type=float, help="Text threshold"
    )
    parser.add_argument(
        "--box-threshold", default=0.4, type=float, help="Box threshold"
    )
    parser.add_argument("--filter-threshold", default=200, type=int)
    
    
    parser.add_argument("--annotator", choices=["GSAM", "VLM"], help="Select the method of annotation")
    parser.add_argument("--images_to_sample", default=10, type=int, help="Number of images to sample from the video for input to VLM")
    parser.add_argument("--gsam_text", nargs="+", help="List of text")
    parser.add_argument("--tgt_obj_text", nargs="+", help="Describe the target object text. Cannot be used with --text")
    parser.add_argument("--ref_obj_text", nargs="+", help="Describe the reference object text. Cannot be used with --text")
    

    # Mesh generation args
    parser.add_argument("--mask_idx", type=int, help="Mask index to generate mesh for", default=1)
    parser.add_argument("--frame_idx", type=int, help="Which frame to use for mesh generation", default=0)
    parser.add_argument("--mask_rembg", type=int, help="Whether to remove background from the mask", default=1)
    parser.add_argument("--library_rembg", type=int, help="Whether to use the rembg library to remove the background", default=1)

    

    return parser.parse_args()

def run_commands(args):
    start_time = time.time()
    orion_env = "ORION"
    video = args.video
    steps_to_skip = args.skip_specific_steps
    for i in range(len(steps_to_skip)):
        steps_to_skip[i] = int(steps_to_skip[i])
    print(f"Skipping steps {steps_to_skip}")
    step = 0

    # 0 - Turn image into hdf5
    print("===================== 0 - Turning video into hdf5 =====================")

    # check if folder with name of video exists
    video_name = os.path.basename(video).split(".")[0]
    if args.output_folder_name:
        output_folder_name = args.output_folder_name
    else:
        output_folder_name = video_name
        
    to_check = "/home/arisrei/research/RPL/ORION/datasets/iphones"
    if step in steps_to_skip or (args.skip_ran_steps and os.path.exists(os.path.join(to_check, video_name))):
        print(f"Folder {video_name} already exists in {to_check}, skipping #0")
    else:
        video_to_hdf5 = "./scripts/extension_work/video_to_hdf5.py"
        command = (
            f"conda run -n {orion_env} " +
            f"python {video_to_hdf5} " +
            f"--video-path {video} " +
            f"--dust3r-intrinsics"
        )
        
        print("Running command:", command)
        subprocess.run(command, shell=True, check=True)

    step += 1
    print("===================== Finished running command #0 =====================\n\n")


    # 1 - GSAM Annotation
    print("===================== 1 - GSAM/VLM Annotation =====================")
    gsam_annotation = "/home/arisrei/research/RPL/ORION/scripts/01_gam_annotation.py"
    annotation_str = ""
    
    if args.annotator == "GSAM":
        if args.gsam_text:
            gsam_text_str = ""
            for arg in args.gsam_text:
                gsam_text_str += f"'{arg}' "
            gsam_text_str = gsam_text_str.strip()
            annotation_str = f"--text {gsam_text_str}"
        elif args.tgt_obj_text and args.ref_obj_text:
            tgt_obj_text_str = ""
            for arg in args.tgt_obj_text:
                tgt_obj_text_str += f"'{arg}' "
            tgt_obj_text_str = tgt_obj_text_str.strip()
            
            ref_obj_text_str = ""
            for arg in args.ref_obj_text:
                ref_obj_text_str += f"'{arg}' "
            ref_obj_text_str = ref_obj_text_str.strip()
            
            annotation_str = f"--tgt_obj_text {tgt_obj_text_str} --ref_obj_text {ref_obj_text_str}"
        else:
            raise ValueError("Either --gsam_text or --tgt_obj_text and --ref_obj_text must be provided")
    
    command = (
        f"conda run -n {orion_env} " +
        f"python {gsam_annotation} " +
        f"--human_demo ./datasets/iphones/{video_name}/{video_name}.hdf5 " + 
        f"{annotation_str} " +
        f"--output_folder_name {output_folder_name} " +
        f"--no-bgr " + 
        f"--annotator {args.annotator} " +
        f"--text-threshold {args.text_threshold} " +
        f"--box-threshold {args.box_threshold} " +
        f"--filter-threshold {args.filter_threshold} " + 
        f"--images_to_sample {args.images_to_sample} " +
        f"--debug {args.debug}"
    )

    if step in steps_to_skip or (args.skip_ran_steps and os.path.exists(f"./annotations/human_demo/{output_folder_name}")):
        print(f"Folder {output_folder_name} already exists in ./annotations/human_demo/, skipping #1")
    else:
        print("Running command:", command)
        subprocess.run(command, shell=True, check=True)

    step += 1
    print("===================== Finished running command #1 =====================\n\n")

    # 2 - XMem Annotation
    print("===================== 2 - XMem Annotation =====================")
    xmem_annotation = "/home/arisrei/research/RPL/ORION/scripts/02_preprocess_video.py"
    command = (
        f"conda run -n {orion_env} " +
        f"python {xmem_annotation} " +
        f"--annotation-folder ./annotations/human_demo/{output_folder_name} " +
        f"--save-video " +
        "--no-bgr "
    )

    if step in steps_to_skip or (args.skip_ran_steps and os.path.exists(f"./annotations/human_demo/{output_folder_name}/masks.npz")):
        print(f"File masks.npz already exists in ./annotations/human_demo/{output_folder_name}/, skipping #2")
    else:
        print("Running command:", command)
        subprocess.run(command, shell=True, check=True)
    
    step += 1
    print("===================== Finished running command #2 =====================\n\n")


    # Create mask index to object mapping
    print("===================== Creating mask index to object mapping =====================")    
    if args.skip_ran_steps and os.path.exists(f"./annotations/human_demo/{output_folder_name}/index_to_mask.png"):
        print(f"File index_to_mask.png already exists in ./annotations/human_demo/{output_folder_name}/, skipping mask_to_image_mapping")
    else:
        mask_to_image_mapping(output_folder_name)

    # 3 - FoundationPose formatting
    print("===================== 3 - FoundationPose Formatting =====================")
    if step in steps_to_skip or (args.skip_ran_steps and os.path.exists(f"./annotations/human_demo/{output_folder_name}/FoundationPose")):
        print(f"Folder FoundationPose already exists in ./annotations/human_demo/{output_folder_name}/, skipping #3")
    else:
        create_foundationpose_formatting(annotation_folder=f"./annotations/human_demo/{output_folder_name}")
    
    step += 1
    print("===================== Finished running command #3 =====================\n\n")

    # 4 - Depth estimation
    print("===================== 4 - Depth Estimation =====================")
    depth_estimation_folder = "./third_party/metric3d"
    depth_estimation = os.path.join(depth_estimation_folder, "mono/tools/test_scale_cano.py")
    
    command = (
        f"conda run -n metric3d " +
        f"python {depth_estimation} " +
        "--save-to-fp-dir " + 
        f"--load-from {os.path.abspath(os.path.join(depth_estimation_folder,'weight/metric_depth_vit_giant2_800k.pth'))} " +
        "--launcher None " +
        f"--test_data_path {os.path.abspath(f'./annotations/human_demo/{output_folder_name}/FoundationPose/rgb')} " +
        f"--intrinsic_path {os.path.abspath(f'./annotations/human_demo/{output_folder_name}/FoundationPose/cam_K.txt')} "+
        "--batch_size 2 " +
        f"'mono/configs/HourglassDecoder/vit.raft5.giant2.py'"
    )

    if step in steps_to_skip or (args.skip_ran_steps and os.path.exists(f"./annotations/human_demo/{output_folder_name}/FoundationPose/depth/all_pred_depth.npy")):
        print(f"Folder depth already exists in ./annotations/human_demo/{output_folder_name}/FoundationPose/, skipping #4")
    else:
        print("Running command:", command)
        subprocess.run(command, shell=True, check=True)

    step += 1
    print("===================== Finished running command #4 =====================\n\n")

    # 5 - Mesh generation
    print("===================== 5 - Mesh Generation =====================")
    if step in steps_to_skip:
        print(f"Skipping #5")
    else:
        print("Creating transparent image with mask")
        create_transparent_bg_from_mask_annotation_folder(f"./annotations/human_demo/{output_folder_name}", args.mask_idx, args.frame_idx)

        print("Running mesh generation")
        mesh_generation = "run.py"
        mesh_generation_img = os.path.abspath(f"./images_for_meshes/{output_folder_name}/{output_folder_name}_with_bg.png")

        no_rembg_str = ""
        if args.mask_rembg:
            mesh_generation_img = os.path.abspath(f"./images_for_meshes/{output_folder_name}/{output_folder_name}_rmbg.png")


        if not args.library_rembg:
            no_rembg_str = "--no_rembg"
            
        
        #mesh_generation_img = os.path.abspath(f"./images_for_meshes/{video_name}/{video_name}_full_rmbg.png")

        command = (
            "cd ./third_party/InstantMesh \n"
            f"conda run -n instantmesh " +
            f"python {mesh_generation} " +
            "./configs/instant-mesh-large.yaml " +
            f"{mesh_generation_img} " +
            #"--save_video " +
            "--export_texmap " + 
            f"{no_rembg_str} " +
            "--save_to_mesh_folder "
        )
        print("Running command:", command)
        subprocess.run(command, shell=True, check=True)
    step += 1
    print("===================== Finished running command #5 =====================\n\n")

    print(f"Total time taken: {time.time() - start_time}")







def mask_to_image_mapping(output_folder_name):
    mask = np.load(f"./annotations/human_demo/{output_folder_name}/masks.npz")['arr_0'][0]
    rgb = np.array(Image.open(f"./annotations/human_demo/{output_folder_name}/frame.jpg"))
    indices = np.unique(mask)
    masked_objects = []
    for i in indices:
        if i == 0:
            continue
        masked_object = rgb.copy()
        masked_object[mask != i] = 0
        masked_objects.append((i, masked_object))

    # Create an image with two columns
    font = ImageFont.load_default(32)
    max_height = max(obj.shape[0] for _, obj in masked_objects)
    total_height = sum(obj.shape[0] for _, obj in masked_objects)
    width = max(obj.shape[1] for _, obj in masked_objects) + 50 # 50 for padding
    index_to_mask = Image.new('RGB', (width, total_height))

    y_offset = 0
    for index, masked_object in masked_objects:
        mask_index_image = Image.new('RGB', (50, masked_object.shape[0]), color=(255, 255, 255))
        draw = ImageDraw.Draw(mask_index_image)
        draw.text((25, masked_object.shape[0] // 2), str(index), fill=(0, 0, 0), font=font)
         
        mask_index_pil = mask_index_image
        masked_object_pil = Image.fromarray(masked_object)
        
        index_to_mask.paste(mask_index_pil, (0, y_offset))
        index_to_mask.paste(masked_object_pil, (50, y_offset))
        
        y_offset += masked_object.shape[0]

    index_to_mask.save(f"./annotations/human_demo/{output_folder_name}/index_to_mask.png")
    print(f"Output image saved at ./annotations/human_demo/{output_folder_name}/index_to_mask.png\n\n")

def main():
    args = parse_args()
    run_commands(args)

if __name__ == "__main__":
    print("\033[91mNote: This assumes that we are using internet videos. So we use dust3r to estimate the intrinsics and metric3d to get estimated depth.\033[0m")
    main()
