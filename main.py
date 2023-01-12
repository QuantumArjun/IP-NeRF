from data import NerfDataset
#Step 1 - Get Camera Poses from Images


class Trainer():
    pass



if __name__ == "__main__":
    nerf_dataset = NerfDataset("data/lego", split_type="train")
    print(len(nerf_dataset))
    print(nerf_dataset[0])