import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import lr_scheduler
from torchmetrics.functional import accuracy, confusion_matrix, f1_score
from raiv_libraries.image_tools import ImageTools


# --- PYTORCH LIGHTNING MODULE ----
class Cnn(pl.LightningModule):

    def __init__(self,
                 courbe_folder=None,
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
            self.train_file = open(courbe_folder + '/train/data_model_train1.txt',
                           'w')  # fichier texte où sont stockées les données des graph (loss, accuracy etc...)
            self.val_file = open(courbe_folder + '/val/data_model_val1.txt', 'w')

    # training loop
    def training_step(self, batch, batch_idx):
        logits, y = self.get_logits_and_outputs(batch)
        return self._calculate_step_metrics(logits, y)

    def training_epoch_end(self, outputs):
        """Compute and log training loss and accuracy at the epoch level."""
        loss_mean = self._calculate_epoch_metrics(outputs, name='Train')
        self.log(f"\nEpoch {self.current_epoch} : training_epoch_end : loss_mean = ", loss_mean.item())

    # validation loop
    def validation_step(self, batch, batch_idx):
        logits, y = self.get_logits_and_outputs(batch)
        # Compute loss & metrics:
        outputs = self._calculate_step_metrics(logits, y)
        self.log("val_loss", outputs["loss"])
        return outputs

    def validation_epoch_end(self, outputs):
        """Compute and log validation loss and accuracy at the epoch level."""
        loss_mean = self._calculate_epoch_metrics(outputs, name='Val')
        self.log("==> validation_epoch_end : loss_mean = ", loss_mean.item())

    # test loop
    def test_step(self, batch, batch_idx):
        logits, y = self.get_logits_and_outputs(batch)
        # Compute loss & metrics:
        return self._calculate_step_metrics(logits, y)

    def test_epoch_end(self, outputs):
        loss_mean = self._calculate_epoch_metrics(outputs, name='Test')
        self.log("test_epoch_end : loss_mean = ", loss_mean.item())

    # define optimizers
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9)
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
    @staticmethod
    def _loss_function(logits, labels):
        weights = torch.tensor([7.0, 3.0]).to(logits.device)#.cuda()
        loss = F.cross_entropy(logits, labels, weight=weights, reduction='mean')
        return loss

    # TODO: Refactor internal metrics
    def _calculate_step_metrics(self, logits, y):
        # prepare the metrics
        loss = self._loss_function(logits[1], y)
        # loss = F.cross_entropy(logits[1], y)
        preds = torch.argmax(logits[1], dim=1)
        num_correct = torch.eq(preds.view(-1), y.view(-1)).sum()
        acc = accuracy(preds, y)
        f1score = f1_score(preds, y, num_classes=2, average='weighted')
        cm = confusion_matrix(preds, y, num_classes=2, )
        return {'loss': loss,
                'acc': acc,
                'f1_score': f1score,
                'confusion_matrix': cm,
                'num_correct': num_correct}

    def _calculate_epoch_metrics(self, outputs, name):
        # Logging activations
        loss_mean = torch.stack([output['loss']
                                 for output in outputs]).mean()
        acc_mean = torch.stack([output['num_correct']
                                for output in outputs]).sum().float()
        acc_mean /= (len(outputs) * self.hparams.batch_size)
        f1score = torch.stack([output['f1_score']
                                for output in outputs]).mean()
        #Text writing
        if self.hparams.courbe_folder:
            if name == 'Train' :
                txt = '\n' + str(self.current_epoch)
                self.train_file.write(txt)
                txt2 = ';' + str(loss_mean.item()) + ';' + str(acc_mean.item()) + ';' + str(f1score.item())
                self.train_file.write(txt2)
            if name == 'Val' :
                txt = '\n' + str(self.current_epoch)
                self.val_file.write(txt)
                txt2 = ';' + str(loss_mean.item()) + ';' + str(acc_mean.item()) + ';' + str(f1score.item())
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
        return loss_mean
