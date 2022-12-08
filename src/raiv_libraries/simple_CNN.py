
from typing import Generator
import torch
from datetime import datetime
import numpy as np
import cv2
from torch.nn import Module
import torch.nn.functional as F

from raiv_libraries.image_tools import ImageTools
from torchvision.models import ResNet18_Weights
from raiv_libraries.CNN import CNN


# --- PYTORCH LIGHTNING MODULE ----
class Simple_CNN(CNN):

    # defines the network
    def __init__(self,
                 courbe_folder = None,
                 learning_rate: float = 1e-3,
                 batch_size: int = 8,
                 input_shape: list = [3, ImageTools.IMAGE_SIZE_FOR_NN, ImageTools.IMAGE_SIZE_FOR_NN],
                 backbone: str = 'resnet18',
                 train_bn: bool = True,
                 milestones: tuple = (5, 10),
                 lr_scheduler_gamma: float = 1e-1):

        super(Simple_CNN, self).__init__()
        # parameters
        self.save_hyperparameters()
        self.dim = input_shape
        # 'vgg16', 'resnet50', 'alexnet', 'resnet18', 'resnet34', 'squeezenet1_1', 'googlenet'
        self.backbone = backbone
        self.train_bn = train_bn
        self.milestones = milestones
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.courbe_folder = courbe_folder
        # self.lr = config["lr"]
        # self.batch_size = config["batch_size"]
        # build the model with one CNN feature extractor using RGB images
        self.fc = self.build_model(['rgb'])

    # mandatory
    def forward(self, t):
        """Forward pass. Returns logits."""
        # 1. Feature extraction:
        t = self.feature_extractors[0](t)
        # print("t:", t.size())
        features = t.squeeze(-1).squeeze(-1)
        # print("Features", features.size())
        # 2. Classifier (returns logits):
        t = self.fc(features)
        # We want the probability to sum 1
        t = F.log_softmax(t, dim=1)
        return features, t


    # trainning loop
    def training_step(self, batch, batch_idx):
        # x = images , y = batch, logits = labels
        x, y = batch
        logits = self(x)
        # 2. Compute loss & metrics:
        return self._calculate_step_metrics(logits, y)

    # validation loop
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # 2. Compute loss & metrics:
        outputs = self._calculate_step_metrics(logits, y)
        self.log("val_loss", outputs["loss"])
        return outputs

    # test loop
    def test_step(self, batch, batch_idx):
        x, y = batch
        print('Shape of X', x.shape)
        print('Shape of y', y.shape)
        nb_img = len(x)
        for idx in np.arange(nb_img):
            img = ImageTools.inv_trans(x[idx])
            npimg = img.cpu().numpy()
            npimg = npimg*256
            npimgt = np.transpose(npimg, (1, 2, 0))
            image_name = str(datetime.now()) + '_' + str(idx + 1) + '.png'
            image_path = '/common/stockage_image_test/' + image_name
            img_rgb = cv2.cvtColor(npimgt, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_path, img_rgb)
        logits = self(x)
        # 2. Compute loss & metrics:
        return self._calculate_step_metrics(logits, y)
