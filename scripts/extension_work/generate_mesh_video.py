import trimesh
import numpy as np
import argparse
from tqdm import tqdm
import io
from PIL import Image
import imageio
import os
from xvfbwrapper import Xvfb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_loc", type=str)
    parser.add_argument("--out_loc", type=str)
    parser.add_argument("--num_frames", type=int, default=30)
    args = parser.parse_args()


    
    # Load your mesh (replace 'your_mesh_file.obj' with your actual file)
    mesh = trimesh.load_mesh(args.mesh_loc)

    #rotate mesh on x axis
    #mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))

    # Define the number of frames for the full rotation
    num_frames = args.num_frames

    # Define the center and radius for the camera orbit
    center = mesh.centroid
    radius = 2 * np.max(mesh.bounding_box.extents)

    # Create a scene from the mesh
    scene = mesh.scene()

    vdisplay = Xvfb()
    vdisplay.start()

    # Generate frames
    frames = []
    frames_np = []
    for i in tqdm(range(num_frames)):
        angle = 2 * np.pi * i / num_frames
        camera_position = center + radius * np.array([np.cos(angle), 0, np.sin(angle)])
        scene.set_camera(angles=(0, angle, 0), distance=1.2 * radius, center=center)

        # Render the scene to an image
        frame = scene.save_image(resolution=(800, 600))
        frames.append(frame)
        frames_np.append(np.array(Image.open(io.BytesIO(frame))))

    vdisplay.stop()

    frames_np = np.array(frames_np)

    mesh_name = os.path.basename(args.mesh_loc).split('.')[0]

    if args.out_loc is None:
        args.out_loc = os.path.dirname(args.mesh_loc)
    writer = imageio.get_writer(os.path.join(args.out_loc, f'{mesh_name}.mp4'), fps=15)
    for i in range(frames_np.shape[0]):
        writer.append_data(frames_np[i])
    writer.close()
    print("saved to", os.path.join(args.out_loc, f'{mesh_name}.mp4'))

if __name__ == '__main__':
    main()