import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import torchvision
from torch.optim import lr_scheduler
from torchmetrics.functional import accuracy, confusion_matrix, f1_score
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from raiv_libraries.image_tools import ImageTools
from pytorch_lightning.loggers import TensorBoardLogger
from ray.tune.integration.pytorch_lightning import TuneReportCallback
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import datetime
import io
import seaborn as sn
import pandas as pd
from PIL import Image

plt.switch_backend('Qt5Agg')
torch.set_printoptions(linewidth=120)


# --- PYTORCH LIGHTNING MODULE ----
class Cnn(pl.LightningModule):

    def __init__(self,  config, courbe_folder=None,
                 learning_rate: float = 1e-3,
                 batch_size: int = 8,
                 input_shape: list = [3, ImageTools.IMAGE_SIZE_FOR_NN, ImageTools.IMAGE_SIZE_FOR_NN],
                 backbone: str = 'resnet18',
                 train_bn: bool = True,
                 milestones: tuple = (5, 10),
                 lr_scheduler_gamma: float = 1e-1):
        super(Cnn, self).__init__()
        self.save_hyperparameters()
        self.build_model()
        if courbe_folder is not None:
            self.train_file = open(courbe_folder + '/train/data_model_train1.txt', 'w')  # fichier texte où sont stockées les données des graph (loss, accuracy etc...)
            self.val_file = open(courbe_folder + '/val/data_model_val1.txt', 'w')

    def build_trainer(self, data_module, model_name, ckpt_dir, num_epochs, suffix, dataset_size):
        """
        Build the Pytorch trainer and a Tensorflow Board
        """
        self.MODEL_CKPT_PATH = Path(ckpt_dir)
        now = datetime.datetime.now()
        filename = f'model_{now.year}_{now.month}_{now.day}-{now.hour}_{now.minute}'
        if dataset_size is not None:
            filename = filename + '-' + str(dataset_size) + '_images'
        if suffix != '':
            filename = filename + '_' + suffix
        self.MODEL_CKPT = self.MODEL_CKPT_PATH / model_name / filename
        # Tensorboard Logger used
        logger = TensorBoardLogger('runs', name=f'Model_{model_name}')
        # # Samples required by the custom ImagePredictionLogger callback to log image predictions.
        # val_samples = next(iter(data_module.val_dataloader()))
        # # ImageTools.show_image(val_samples[0])  # ImageTools.show_image(val_samples[0][0]) to show only the first image (not all images in the batch)
        # grid = ImageTools.inv_trans(torchvision.utils.make_grid(val_samples[0], nrow=8, padding=2))
        # # write to tensorboard
        # logger.experiment.add_image('test', grid)
        # logger.finalize("success")
        # Load callbacks ########################################
        checkpoint_callback, early_stop_callback = self._config_callbacks()
        # Trainer  ################################################
        metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
        return pl.Trainer(max_epochs=num_epochs,
                             devices="auto", accelerator="auto",
                             auto_select_gpus=False,
                             auto_lr_find=True,
                             logger=logger,
                             log_every_n_steps=10,
                             # callbacks=[early_stop_callback, checkpoint_callback])
                             callbacks=[checkpoint_callback, TuneReportCallback(metrics, on="validation_end")])

    # training loop
    def training_step(self, batch, batch_idx):
        logits, y = self.get_logits_and_outputs(batch)
        return self._calculate_step_metrics(logits, y)

    def training_epoch_end(self, outputs):
        """Compute and log training loss and accuracy at the epoch level."""
        loss_mean, acc_mean = self._calculate_epoch_metrics(outputs, name='Train')

    # validation loop
    def validation_step(self, batch, batch_idx):
        logits, y = self.get_logits_and_outputs(batch)
        # Compute loss & metrics:
        outputs = self._calculate_step_metrics(logits, y)
        outputs["pred_out"]=logits[1]  # Predicted outputs
        outputs["true_out"]=y  # Real outputs
        self.log("val_loss", outputs["loss"])
        return outputs

    def validation_epoch_end(self, outputs):
        """Compute and log validation loss and accuracy at the epoch level."""
        loss_mean, acc_mean = self._calculate_epoch_metrics(outputs, name='Val')
        self.log("ptl/val_loss", loss_mean)
        self.log("ptl/val_accuracy", acc_mean)
        # Generate a convolution matrix for TensorBoard
        tb = self.logger.experiment  # noqa
        pred_out = torch.cat([tmp['pred_out'] for tmp in outputs])
        true_out = torch.cat([tmp['true_out'] for tmp in outputs])
        confusion = torchmetrics.ConfusionMatrix(num_classes=2).to(pred_out.get_device())
        confusion(pred_out, true_out)
        computed_confusion = confusion.compute().detach().cpu().numpy().astype(int)
        # confusion matrix
        df_cm = pd.DataFrame(
            computed_confusion,
            ['fail', 'success'],  # Lines
            ['fail', 'success'],  # Columns
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.subplots_adjust(left=0.05, right=.65)
        sn.set(font_scale=1.2)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d', ax=ax)
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg', bbox_inches='tight')
        buf.seek(0)
        im = Image.open(buf)
        im = torchvision.transforms.ToTensor()(im)
        tb.add_image("val_confusion_matrix", im, global_step=self.current_epoch)


    # test loop
    def test_step(self, batch, batch_idx):
        logits, y = self.get_logits_and_outputs(batch)
        # Compute loss & metrics:
        return self._calculate_step_metrics(logits, y)

    def test_epoch_end(self, outputs):
        loss_mean, acc_mean = self._calculate_epoch_metrics(outputs, name='Test')

    # define optimizers
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.config['learning_rate'], momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return (
            {'optimizer': optimizer, 'lr_scheduler': scheduler}
        )

    def get_cumulative_output_conv_layers_size(self, feature_extractors):
        """
        Return the cumulative size of all the output convolution layers which is the input size for the dense part
        """
        batch_size = 1
        input_data = torch.autograd.Variable(torch.rand(batch_size, *self.hparams.input_shape))
        size = 0
        for feature_extractor in feature_extractors:
            output_feat = feature_extractor(input_data)  # returns the feature tensor from the conv block
            n_size = output_feat.data.view(batch_size, -1).size(1)  #the size of the output tensor going into the Linear layer from the conv block
            size += n_size
        return size

    # loss function, weights modified to give more importance to class 1
    def _loss_function(self, logits, labels):
        weights = torch.tensor([7.0, 3.0]).to(logits.device)#.cuda()
        loss = F.cross_entropy(logits, labels, weight=weights, reduction='mean')
        return loss


    def _config_callbacks(self):
        # Checkpoint  ################################################
        # Saves the models so it is possible to access afterwards
        checkpoint_callback = ModelCheckpoint(dirpath=str(self.MODEL_CKPT_PATH),
                                              filename=str(self.MODEL_CKPT),
                                              monitor='val_loss',
                                              save_top_k=1,
                                              mode='min',
                                              save_weights_only=True)
        # EarlyStopping  ################################################
        # Monitor a validation metric and stop training when it stops improving.
        early_stop_callback = EarlyStopping(monitor='val_loss',
                                            min_delta=0.0,
                                            patience=5,
                                            verbose=False,
                                            mode='min')
        # tune_report_callback = TuneReportCallback({"loss": "ptl/val_loss",
        #                                            "mean_accuracy": "ptl/val_accuracy"}, on="validation_end")
        return checkpoint_callback, early_stop_callback

    # TODO: Refactor internal metrics
    def _calculate_step_metrics(self, logits, y):
        # prepare the metrics
        loss = self._loss_function(logits[1], y)
        # loss = F.cross_entropy(logits[1], y)
        preds = torch.argmax(logits[1], dim=1)
        num_correct = torch.eq(preds.view(-1), y.view(-1)).sum()
        acc = accuracy(preds, y)
        f1score = f1_score(preds, y, num_classes=2, average='weighted')
        return {'loss': loss,
                'acc': acc,
                'f1_score': f1score,
                'num_correct': num_correct,
                'total': len(preds)}

    def _calculate_epoch_metrics(self, outputs, name):
        # Logging activations
        loss_mean = torch.stack([output['loss']
                                 for output in outputs]).mean()
        correct = sum([x['num_correct'] for x in outputs])
        total = sum([x['total'] for x in outputs])
        acc_mean = correct / total
        f1score = torch.stack([output['f1_score']
                                for output in outputs]).mean()
        #Text writing
        if self.hparams.courbe_folder:
            txt = '\n' + str(self.current_epoch)
            txt2 = ';' + str(loss_mean.item()) + ';' + str(acc_mean.item()) + ';' + str(f1score.item())
            if name == 'Train' :
                self.train_file.write(txt)
                self.train_file.write(txt2)
            if name == 'Val' :
                self.val_file.write(txt)
                self.val_file.write(txt2)
        # Logging scalars
        self.logger.experiment.add_scalar(f'Loss/{name}',
                                          loss_mean,
                                          self.current_epoch)
        self.logger.experiment.add_scalar(f'Accuracy/{name}',
                                          acc_mean,
                                          self.current_epoch)
        self.logger.experiment.add_scalar(f'F1_Score/{name}',
                                          f1score,
                                          self.current_epoch)
        return loss_mean, acc_mean


    def _plot_classes_preds(self, images, labels):
        '''
        Generates matplotlib Figure using a trained network, along with images
        and labels from a batch, that shows the network's top prediction along
        with its probability, alongside the actual label, coloring this
        information based on whether the prediction was correct or not.
        Uses the "images_to_probs" function.
        '''
        preds, probs = self._images_to_probs(images)
        # plot the images in the batch, along with predicted and true labels
        my_dpi = 96 # For my monitor (see https://www.infobyip.com/detectmonitordpi.php)
        nb_images = len(images)
        fig = plt.figure(figsize=(nb_images * ImageTools.IMAGE_SIZE_FOR_NN/my_dpi, ImageTools.IMAGE_SIZE_FOR_NN/my_dpi), dpi=my_dpi)
        class_names = self.image_module._find_classes()
        for idx in np.arange(nb_images):
            ax = fig.add_subplot(1, nb_images, idx + 1, xticks=[], yticks=[])
            img = ImageTools.inv_trans(images[idx])
            npimg = img.cpu().numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                class_names[preds[idx]],
                probs[idx] * 100.0,
                class_names[labels[idx]]),
                color=("green" if preds[idx] == labels[idx].item() else "red"))
        return fig


    def _images_to_probs(self, images):
        '''
        Generates predictions and corresponding probabilities from a trained
        network and a list of images
        '''
        output = self.model(images)
        # convert output probabilities to predicted class
        _, preds_tensor = torch.max(output[1], 1)
        # preds = np.squeeze(preds_tensor.cpu().numpy())  CAusait une erreur quand il n'y avait qu'une seule image
        preds = preds_tensor.cpu().numpy()
        return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output[1])]

    # Static methods

    @staticmethod
    def compute_prob_and_class(pred):
        """ Retrieve class (success or fail) and its associated percentage [0,1] from pred """
        prob, cl = torch.max(pred, 1)
        if cl.item() == 0:  # Fail
            prob = 1 - prob.item()
        else:  # Success
            prob = prob.item()
        return prob, cl