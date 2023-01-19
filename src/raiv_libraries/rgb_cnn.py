import torch
import torch.nn.functional as F
import torchvision.models as models
from raiv_libraries.cnn import Cnn

from raiv_libraries.image_tools import ImageTools


class RgbCnn(Cnn):

    def __init__(self, **kwargs):
        super(RgbCnn, self).__init__(**kwargs)

    def build_model(self):
        """ Define model layers """
        # Load pre-trained network: choose the model for the pretrained network
        model_func = getattr(models, self.hparams.backbone)
        backbone = model_func(weights="DEFAULT")
        # Feature extractor
        _layers = list(backbone.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*_layers)
        # classes are two: success or failure
        num_target_classes = 2
        n_sizes = self.get_cumulative_output_conv_layers_size([self.feature_extractor])
        # Classifier
        _fc_layers = [torch.nn.Linear(n_sizes, 256),
                      torch.nn.Linear(256, 32),
                      torch.nn.Linear(32, num_target_classes)]
        self.fc = torch.nn.Sequential(*_fc_layers)

    def forward(self, t):
        """Forward pass. Returns logits."""
        # 1. Feature extraction:
        t = self.feature_extractor(t)
        # print("t:", t.size())
        features = t.squeeze(-1).squeeze(-1)
        # 2. Classifier (returns logits):
        t = self.fc(features)
        t = F.log_softmax(t, dim=1)
        return features, t

    def get_logits_and_outputs(self, batch):
        rgb, y = batch
        logits = self(rgb)
        return logits, y

    @staticmethod
    @torch.no_grad()
    def predict_from_pil_rgb_image(model, pil_rgb_img):
        image_tensor = ImageTools.image_preprocessing(pil_rgb_img)
        features, prediction = model(image_tensor)
        prediction = prediction.detach()
        return torch.exp(prediction)

    @staticmethod
    def load_ckpt_model_file(ckpt_model_filename):
        """
        Load the model named 'ckpt_model_filename' and freeze it
        :param name: name of the model
        :return: the model freezed to be used for inference
        """
        model = RgbCnn.load_from_checkpoint(ckpt_model_filename)
        model.freeze()
        return model
