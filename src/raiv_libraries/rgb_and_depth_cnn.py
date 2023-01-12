import torch
import torch.nn.functional as F
import torchvision.models as models
from raiv_libraries.cnn import Cnn

from raiv_libraries.image_tools import ImageTools


class RgbAndDepthCnn(Cnn):
    def __init__(self, **kwargs):
        super(RgbAndDepthCnn, self).__init__(**kwargs)

    def build_model(self):
        """Define model layers """
        # Load pre-trained network: choose the model for the pretrained network
        model_func = getattr(models, self.hparams.backbone)
        # Feature extractors
        self.feature_extractor_rgb = self._build_features_layers(model_func)
        self.feature_extractor_depth = self._build_features_layers(model_func)
        # classes are two: success or failure
        num_target_classes = 2
        n_sizes = self.get_cumulative_output_conv_layers_size([self.feature_extractor_rgb, self.feature_extractor_depth])
        # Classifier
        _fc_layers = [torch.nn.Linear(n_sizes, 256),
                      torch.nn.Linear(256, 32),
                      torch.nn.Linear(32, num_target_classes)]
        self.fc = torch.nn.Sequential(*_fc_layers)

    def _build_features_layers(self, model_func):
        """ Return the feature extractor from a pretrained Cnn """
        backbone = model_func(weights="DEFAULT")
        # Feature extractor
        _layers = list(backbone.children())[:-1]
        return torch.nn.Sequential(*_layers)

    def forward(self, rgb, depth):
        """Forward pass. Returns logits."""
        # 1. Feature extraction for RGB Cnn
        features_rgb = self.feature_extractor_rgb(rgb)
        features_rgb = features_rgb.squeeze(-1).squeeze(-1)
        # 2. Feature extraction for Depth Cnn
        features_depth = self.feature_extractor_rgb(depth)
        features_depth = features_depth.squeeze(-1).squeeze(-1)
        # 3. Concatenate both features
        features = torch.cat((features_rgb, features_depth), dim=1)
        # 4. Classifier (returns logits):
        t = self.fc(features)
        t = F.log_softmax(t, dim=1)
        return features, t

    def get_logits_and_outputs(self, batch):
        rgb, depth, y, _ = batch
        logits = self(rgb, depth)
        return logits, y

    @staticmethod
    @torch.no_grad()
    def predict_from_pil_rgb_and_depth_images(model, pil_rgb_img, pil_depth_img):
        rgb_tensor = ImageTools.image_preprocessing(pil_rgb_img)
        depth_tensor = ImageTools.image_preprocessing(pil_depth_img)
        features, prediction = model(rgb_tensor, depth_tensor)
        prediction = prediction.detach()
        return torch.exp(prediction)

    @staticmethod
    def load_ckpt_model_file(ckpt_model_filename):
        """
        Load the model named 'ckpt_model_filename' and freeze it
        :param name: name of the model
        :return: the model freezed to be used for inference
        """
        model = RgbAndDepthCnn.load_from_checkpoint(ckpt_model_filename)
        model.freeze()
        return model