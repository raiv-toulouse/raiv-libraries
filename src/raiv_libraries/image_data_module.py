# import libraries

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision
import torchvision.datasets as datasets
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, random_split, Subset, WeightedRandomSampler
from raiv_libraries.image_tools import ImageTools
import cv2


class ImageDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size, dataset_size=None, num_workers=8):
        super().__init__()
        self.trains_dims = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.dataset_size = dataset_size

    def _calculate_weights(self, dataset):
        class_count = Counter(dataset.targets)
        print("Class fail:", class_count[0])
        print("Class success:", class_count[1])
        count = np.array([class_count[0], class_count[1]])
        weight = 1. / torch.Tensor(count)
        weight_samples = np.array([weight[t] for t in dataset.targets])
        return weight_samples

    def setup(self, stage=None):
        # Build Dataset
        dataset = datasets.ImageFolder(self.data_dir)
        if self.dataset_size is None:
            weight_samples = self._calculate_weights(dataset)
            # Select a subset of the images
            dataset_size = len(dataset) if self.dataset_size is None else min(len(dataset), self.dataset_size)
            samples = list(WeightedRandomSampler(weight_samples, len(weight_samples),
                                                 replacement=False,
                                                 generator=torch.Generator().manual_seed(42)))[:dataset_size]
        else:  # we get self.dataset_size images from 'success' and 'fail' folders
            total_indices = list(range(len(dataset)))
            class_count = Counter(dataset.targets)
            dataset_size = min(self.dataset_size, class_count[0], class_count[1])
            fail_indices = total_indices[:dataset_size]
            success_indices = total_indices[class_count[0]:class_count[0]+dataset_size]
            samples = fail_indices + success_indices
            np.random.shuffle(samples)

        subset = Subset(dataset, indices=samples)

        train_size = int(0.7 * len(subset))
        val_size = int(0.5 * (len(subset) - train_size))
        test_size = int(len(subset) - train_size - val_size)

        train_data, val_data, test_data = random_split(subset,
                                                       [train_size, val_size, test_size],
                                                       generator=torch.Generator().manual_seed(42))

        print("Len Train Data", len(train_data))
        print("Len Val Data", len(val_data))
        print("Len Test Data", len(test_data))

        self.train_data = TransformSubset(train_data, transform=ImageTools.transform_image)
        self.val_data = TransformSubset(val_data, transform=ImageTools.transform_image)
        self.test_data = TransformSubset(test_data, transform=ImageTools.transform_image)



    def train_dataloader(self, num_workers=None):
        return self._generate_dataloader(self.train_data, num_workers)

    def val_dataloader(self, num_workers=None):
        return self._generate_dataloader(self.val_data, num_workers)

    def test_dataloader(self, num_workers=None):
        return self._generate_dataloader(self.test_data, num_workers)

    def _generate_dataloader(self, data, num_workers=None):
        return torch.utils.data.DataLoader(data, num_workers=num_workers if num_workers else self.num_workers, batch_size=self.batch_size)

    # TODO: Método para acceder a las clases
    def _find_classes(self):
        classes = [d.name for d in os.scandir(self.data_dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes

    @staticmethod
    def _count_targets(subset):
        class_0 = 0
        class_1 = 0
        for tensor, target in subset:
            if target == 0:
                class_0 += 1
            else:
                class_1 += 1
        print('Count class 0:', class_0)
        print('Count class 1:', class_1)

    def plot_classes_images(self, images, labels):
        '''
        Generates matplotlib Figure along with images and labels from a batch
        To be used with : writer.add_figure('Title', plot_classes_preds(images, classes))
        '''
        # plot the images in the batch, along with predicted and true labels
        class_names = self._find_classes()
        nb_images = len(images)
        my_dpi = 96  # For my monitor (see https://www.infobyip.com/detectmonitordpi.php)
        fig = plt.figure(figsize=(nb_images * 256 / my_dpi, 256 / my_dpi), dpi=my_dpi)
        for idx in np.arange(nb_images):
            ax = fig.add_subplot(1, nb_images, idx + 1, xticks=[], yticks=[])
            img = ImageTools.inv_trans(images[idx])
            npimg = img.cpu().numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(class_names[labels[idx]])
        return fig


class TransformSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

    def _count_targets(self):
        count_class = [0, 0]
        for tensor, target in self.subset:
            if target == 0:
                count_class[0] += 1
            else:
                count_class[1] += 1
        return count_class


# --- MAIN ----
if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    import argparse

    parser = argparse.ArgumentParser(description='Test ImageDataModule which loads images from specified folder. View results with : tensorboard --logdir=runs')
    parser.add_argument('images_folder', type=str, help='images folder with fail and success sub-folders')
    parser.add_argument('-t', '--test_num_workers', action="store_true", help='if we want to perform a num_workers test')
    parser.add_argument('-d', '--dataset_size', default=None, type=int, help='Optionnal number of images for the dataset size')
    args = parser.parse_args()

    image_module = ImageDataModule(data_dir=args.images_folder, batch_size=8, num_workers=8, dataset_size=args.dataset_size)
    image_module.setup()

    if args.test_num_workers:
        print('test num_workers')
        from time import time
        import multiprocessing as mp

        for num_workers in range(2, mp.cpu_count(), 2):
            train_loader = image_module.train_dataloader(num_workers=num_workers) #DataLoader(train_reader, shuffle=True, num_workers=num_workers, batch_size=64, pin_memory=True)
            start = time()
            for epoch in range(1, 3):
                for i, data in enumerate(train_loader, 0):
                    pass
            end = time()
            print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
    else:
        images, labels = next(iter(image_module.val_dataloader()))
        print(images.shape)
        ImageTools.show_image(images[0])
        print(images[0].shape)
        grid = torchvision.utils.make_grid(images, nrow=8, padding=2)
        writer = SummaryWriter()
        writer.add_figure('images with labels', image_module.plot_classes_images(images, labels))
        writer.add_image('some_images', ImageTools.inv_trans(grid))
        writer.close()


