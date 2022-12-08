

import torch
import torch.nn.functional as F
import torchvision.models as models
from raiv_libraries.image_tools import ImageTools
from raiv_libraries.CNN import CNN

# --- PYTORCH LIGHTNING MODULE ----
class Double_CNN(CNN):

    # defines the network
    def __init__(self,
                 learning_rate: float = 1e-3,
                 batch_size: int = 8,
                 input_shape: list = [3, ImageTools.IMAGE_SIZE_FOR_NN, ImageTools.IMAGE_SIZE_FOR_NN],
                 backbone: str = 'resnet18',
                 train_bn: bool = True,
                 milestones: tuple = (5, 10),
                 lr_scheduler_gamma: float = 1e-1,
                 num_workers: int = 6):

        super(Double_CNN, self).__init__()
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
        self.num_workers = num_workers
        # build the model with two CNN feature extractors using RGB images and depth images
        self.fc = self.build_model(['rgb', 'depth'])
        # self.train_file = open('/common/data_courbes_matplotlib/DEPTH/150im_depth/train/data_model_TRAIN1.txt',
        #                'w')  # fichier texte où sont stockées les données des graph (loss, accuracy etc...)
        # self.val_file = open('/common/data_courbes_matplotlib/DEPTH/150im_depth/val/data_model_VAL1.txt', 'w')

    # mandatory
    def forward(self, rgb, depth):
        """Forward pass. Returns logits."""
        # 1. Feature extraction for RGB CNN
        rgb_feature_extractor = self.feature_extractors[0]
        rgb = rgb_feature_extractor(rgb)
        features_rgb = rgb.squeeze(-1).squeeze(-1)
        # 2. Feature extraction for Depth CNN
        depth_feature_extractor = self.feature_extractors[1]
        depth = depth_feature_extractor(depth)
        features_depth = depth.squeeze(-1).squeeze(-1)
        # 3. Concatenate both features
        features = torch.cat((features_rgb, features_depth), dim=1)
        # 4. Classifier (returns logits):
        t = self.fc(features)
        # We want the probability to sum 1
        t = F.log_softmax(t, dim=1)
        return features, t

    def get_logits_and_outputs(self, batch):
        rgb, depth, y, _ = batch
        logits = self(rgb, depth)
        return logits, y



