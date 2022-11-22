
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
