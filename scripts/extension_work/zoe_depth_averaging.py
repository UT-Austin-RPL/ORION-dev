import h5py
import numpy as np
import matplotlib
import argparse
import os
import json
from PIL import Image

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

    if "data/human_demo/obs/zoe_depth" not in demo_file:
        print("No zoe_depth in the file")
        demo_file.close()
        return

    out_uint16 = demo_file.get("data/human_demo/obs/zoe_depth")[()]
    max_and_mins = demo_file.get("data/human_demo/obs/rescale_max_min")[()]
    demo_file.close()

    old_max = max_and_mins[:,0]
    old_min = max_and_mins[:,1]

    dif = (old_max-old_min).reshape(out_uint16.shape[0],1,1,1)
    old_min_reshaped = old_min.reshape(out_uint16.shape[0],1,1,1)

    rescale_back = (((out_uint16 / 65535.0) * dif) + old_min_reshaped).astype(np.float32)
    rescale_back = rescale_back.reshape((rescale_back.shape[0], rescale_back.shape[1], rescale_back.shape[2])) #squeeze the last dimension

    #print("unmasked zoe depth", np.unique(rescale_back))
    masks = np.load(os.path.join(annotations_folder, "masks.npz"))["arr_0"]
    boat_mask = 1
    
    #change all values in masks numpy array to 0 if they are not equal to 1
    masks = np.where(masks != boat_mask, 0, masks)

    #get size of mask where value = 1
    mask_size = np.sum(masks, axis=(1,2))
    print(mask_size.shape, mask_size[0])
    
    #change all values in zoe_depth to 0 if the corresponding value in masks is
    #0
    rescale_back = np.where(masks == 0, 0, rescale_back)
    #print("masked zoe depth", np.unique(rescale_back))

    average_depths = []
    std_depths = []
    # i=0
    # print(np.shape(rescale_back[i]), np.unique(rescale_back[i]))
    # print(np.shape(rescale_back[i][masks[i] == 1]), np.unique(rescale_back[i][masks[i] == 1]))
    for i in range(rescale_back.shape[0]):
        average_depths.append(np.mean(rescale_back[i][masks[i] == 1]))
        std_depths.append(np.std(rescale_back[i][masks[i] == 1]))
    
    #plot average and std of depths with key
    import matplotlib.pyplot as plt
    plt.plot(average_depths)
    plt.plot(std_depths)
    plt.xlabel("Frame")
    plt.ylabel("Depth")
    plt.legend(["Average", "Standard Deviation"])
    plt.savefig("average_and_std_depths.png")





    

    

if __name__ == "__main__":
    main()

    