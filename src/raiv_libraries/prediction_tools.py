import torchvision.transforms as transforms
from torchvision.transforms.functional import crop
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import torchvision.transforms.functional as F
from raiv_libraries.CNN import CNN
from raiv_libraries.image_model import ImageModel
import torch
import ntpath


class PredictTools:

    def load_model(model_path):
        ckpt_model_name = ntpath.basename(model_path)
        dir_name = ntpath.dirname(model_path)
        image_model = ImageModel(model_name='resnet18', ckpt_dir=dir_name)
        inference_model = image_model.load_ckpt_model_file(ckpt_model_name)
        inference_model.freeze()
        return image_model, inference_model

    def predict(model, img):
        features, preds = model.evaluate_image(img, False)  # No processing
        pred = torch.exp(preds)
        return pred

    def compute_prob_and_class(pred):
        """ Retrieve class (success or fail) and its associated percentage from pred """
        prob, cl = torch.max(pred, 1)
        if cl.item() == 0:  # Fail
            prob = 1 - prob.item()
        else:  # Success
            prob = prob.item()
        return prob












def _load_model(self):
    """
    Load a pretrained 'resnet18' model from a CKPT filename, freezed for inference
    """
    self.model = CNN(backbone='resnet18')
    self.inference_model = self.model.load_from_checkpoint(self.model_name)  # Load the selected model
    self.inference_model.freeze()

def predict_from_point(self, x, y):
    """ Predict probability and class for a cropped image at (x,y) """
    self.predict_center_x = x
    self.predict_center_y = y
    image_cropped = ImageTools.crop_xy(self.image, x, y)
    img = ImageTools.transform_image(image_cropped)  # Get the cropped transformed image
    img = img.unsqueeze(0)  # To have a 4-dim tensor ([nb_of_images, channels, w, h])
    return self.predict(img)

def predict_from_image(self):
    """ Load the images data """
    loaded_image = QFileDialog.getOpenFileName(self, 'Open image', '.', "Model files (*.png)",
                                               options=QFileDialog.DontUseNativeDialog)
    if loaded_image[0]:
        self.image = Image.open(loaded_image[0])
    img = ImageTools.transform_image(self.image)  # Get the loaded images, resize in 256 and transformed in tensor
    img = img.unsqueeze(0)  # To have a 4-dim tensor ([nb_of_images, channels, w, h])
    pred = self.predict(img)
    prob, cl = self.canvas._compute_prob_and_class(pred)
    self.prediction_from_image.setText("La prédiction de l'image chargé est : " + str(prob) + " %")
    print(prob)

def predict(self, img):
    features, preds = self.image_model.evaluate_image(img, False)  # No processing
    return torch.exp(preds)

def compute_map(self, start_coord, end_coord):
    """ Compute a list of predictions and ask the canvas to draw them
        Called from CanvasExplore """
    all_preds = self._compute_all_preds(start_coord, end_coord)
    self.canvas.all_preds = all_preds
    self.canvas.repaint()