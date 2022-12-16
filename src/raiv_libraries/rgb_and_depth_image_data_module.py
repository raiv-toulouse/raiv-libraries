import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, random_split, Subset, WeightedRandomSampler
from raiv_libraries.image_dataset_rgb_depth import ImageDatasetRgbDepth
from raiv_libraries.image_tools import ImageTools


class RgbAndDepthImageDataModule(pl.LightningDataModule):

    def __init__(self, data_dir_rgb, data_dir_depth, batch_size=8, dataset_size=None):
        super().__init__()
        self.dataset = ImageDatasetRgbDepth(data_dir_rgb, data_dir_depth)
        self.batch_size = batch_size

    def setup(self, stage=None):
        train_size = int(0.7 * len(self.dataset))
        val_size = int(0.5 * (len(self.dataset) - train_size))
        test_size = int(len(self.dataset) - train_size - val_size)
        train_set, val_set, test_set = random_split(self.dataset, (train_size, val_size, test_size))
        self.train_data = TransformSubset(train_set, transform=ImageTools.transform)  # No augmentation because it doesn't apply the same random transform to RGB and Depth images
        self.val_data = TransformSubset(val_set, transform=ImageTools.transform)
        self.test_data = TransformSubset(test_set, transform=ImageTools.transform)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_data, num_workers=16, batch_size=self.batch_size)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(self.val_data, num_workers=16, batch_size=self.batch_size)
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(self.test_data, num_workers=16, batch_size=self.batch_size)
        return test_loader


class TransformSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        rgb, depth, y , lst_files = self.subset[index]
        if self.transform:
            rgb = self.transform(rgb)
            depth = self.transform(depth)
        return rgb, depth, y, lst_files

    def __len__(self):
        return len(self.subset)


# --- MAIN ----
if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    import argparse
    from raiv_libraries.image_tools import ImageTools

    parser = argparse.ArgumentParser(description='Test ImageDataModule which loads images (rgb AND depth) from specified folder. View results with : tensorboard --logdir=runs')
    parser.add_argument('images_folder_rgb', type=str, help='images folder of rgb images with fail and success sub-folders')
    parser.add_argument('images_folder_depth', type=str, help='images folder of depth images with fail and success sub-folders')
    args = parser.parse_args()

    image_module = RgbAndDepthImageDataModule(data_dir_rgb=args.images_folder_rgb, data_dir_depth=args.images_folder_depth, batch_size=4)
    image_module.setup()

    data = next(iter(image_module.train_dataloader()))

    print(data[0].shape)
    ImageTools.show_image(data[0], data[3][0], inv_needed=True)
    ImageTools.show_image(data[1], data[3][1], inv_needed=True)

