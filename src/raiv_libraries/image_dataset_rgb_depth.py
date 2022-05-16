import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split
import pathlib

class ImageDatasetRgbDepth(Dataset):

    def __init__(self, rgb_dir, depth_dir):
        rgb_dir = pathlib.Path(rgb_dir)
        rgb_fail = sorted(list((rgb_dir / 'fail').iterdir()))
        self.nb_of_fail = len(rgb_fail)
        rgb_success = sorted(list((rgb_dir / 'success').iterdir()))
        self.rgb_files = [*rgb_fail , *rgb_success]
        depth_dir = pathlib.Path(depth_dir)
        depth_fail = sorted(list((depth_dir / 'fail').iterdir()))
        depth_success = sorted(list((depth_dir / 'success').iterdir()))
        self.depth_files = [*depth_fail, *depth_success]


    def __len__(self):
        """ Return the number of files of the rgb folder"""
        return len(self.rgb_files)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_rgb = Image.open(self.rgb_files[idx])
        image_depth = Image.open(self.depth_files[idx])
        image_depth = image_depth.convert("RGB")  # To have a 3 channels image from a grayscale one

        #print(f"RGB = {self.rgb_files[idx]}")
        #print(f"Depth = {self.depth_files[idx]}")

        class_id = 0 if idx < self.nb_of_fail else 1  # [0 : fail, 1 : success]

        return image_rgb, image_depth, class_id, [str(self.rgb_files[idx]), str(self.depth_files[idx])]


# --- MAIN ----
if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    import argparse
    from raiv_libraries.image_tools import ImageTools

    parser = argparse.ArgumentParser(description='Test ImageDataModule which loads images (rgb AND depth) from specified folder. View results with : tensorboard --logdir=runs')
    parser.add_argument('images_folder_rgb', type=str, help='images folder of rgb images with fail and success sub-folders')
    parser.add_argument('images_folder_depth', type=str, help='images folder of depth images with fail and success sub-folders')
    args = parser.parse_args()

    dataset = ImageDatasetRgbDepth(args.images_folder_rgb, args.images_folder_depth)
    print(dataset)
    rgb, depth, class_id = dataset.__getitem__(4)
    print(rgb, depth, class_id)

    train_size = int(0.80 * len(dataset))
    val_size = int((len(dataset) - train_size) / 2)
    test_size = int((len(dataset) - train_size) / 2)

    train_set, val_set, test_set = random_split(dataset, (train_size, val_size, test_size))

    print(train_set)