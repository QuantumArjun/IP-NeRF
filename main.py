from data import NerfDataset
import torch
#Step 1 - Get Camera Poses from Images


class Trainer():
    pass



if __name__ == "__main__":
    nerf_dataset = NerfDataset("data/lego", split_type="train")
    print(len(nerf_dataset))
    print(torch.max(nerf_dataset[0]))
    #TODO: We now have fov and focal distance https://github.com/yenchenlin/nerf-pytorch/issues/41
    # So now use trig to calculate the width of the image plane, and then interpolate for each pixel the ray vector shooting out