from PIL import Image
import torch
from torch.utils.data import Dataset
import pathlib


class RgbAndDepthImageDataset(Dataset):

    def __init__(self, rgb_dir, depth_dir):
        rgb_dir = pathlib.Path(rgb_dir)
        rgb_fail = sorted(list((rgb_dir / 'fail').iterdir()))
        self.nb_of_fail = len(rgb_fail)
        rgb_success = sorted(list((rgb_dir / 'success').iterdir()))
        nb_of_success = len(rgb_success)
        self.targets = [0]*self.nb_of_fail + [1]*nb_of_success # To have the same attribut that ImageFolder have
        self.rgb_files = [*rgb_fail , *rgb_success]
        depth_dir = pathlib.Path(depth_dir)
        depth_fail = sorted(list((depth_dir / 'fail').iterdir()))
        depth_success = sorted(list((depth_dir / 'success').iterdir()))
        self.depth_files = [*depth_fail, *depth_success]

    def __len__(self):
        """ Return the number of files of the rgb folder (it's the same for depth images """
        return len(self.rgb_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_rgb = Image.open(self.rgb_files[idx])
        image_depth = Image.open(self.depth_files[idx])
        image_depth = image_depth.convert("RGB")  # To have a 3 channels image from a grayscale one
        class_id = 0 if idx < self.nb_of_fail else 1  # [0 : fail, 1 : success]
        return image_rgb, image_depth, class_id, [str(self.rgb_files[idx]), str(self.depth_files[idx])]


# --- MAIN ----
if __name__ == '__main__':
    import argparse
    import random

    parser = argparse.ArgumentParser(description='Test RgbAndDepthImageDataset which loads images (rgb AND depth) from specified folder.')
    parser.add_argument('images_folder_rgb', type=str, help='images folder of rgb images with fail and success sub-folders')
    parser.add_argument('images_folder_depth', type=str, help='images folder of depth images with fail and success sub-folders')
    args = parser.parse_args()

    dataset = RgbAndDepthImageDataset(args.images_folder_rgb, args.images_folder_depth)
    nb_files = dataset.__len__()
    print(f"Number of files = {nb_files}")
    ind = random.randint(0, nb_files)-1
    rgb, depth, class_id, files = dataset.__getitem__(ind)
    print(f"class : {class_id}", files)
    rgb.show()
    depth.show()
