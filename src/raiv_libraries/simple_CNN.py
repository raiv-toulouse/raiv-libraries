import torch
import torch.nn.functional as F
import torchvision.models as models
from raiv_libraries.CNN import CNN


class Simple_CNN(CNN):

    def __init__(self, **kwargs):
        super(Simple_CNN, self).__init__(**kwargs)

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
        x, y = batch
        logits = self(x)
        return logits, y
