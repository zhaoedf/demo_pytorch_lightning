import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torch.nn as nn 

from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy

from utils import Dice_coeff

# TODO check self.log epoch/step
class SegLearner(pl.LightningModule):

    # --------------- computations ---------------
    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore='model')
        
        self.loss_func = nn.BCEWithLogitsLoss()
        self.dice_coeff = Dice_coeff()


    # --------------- training loop ---------------
    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']

        logits = self.model(x)  # probs = sigmoid(logits)
        y_hat = torch.sigmoid(logits)
        
        if isinstance(self.loss_func, nn.BCEWithLogitsLoss):
            loss = self.loss_func(logits, y)
        else:
            loss = self.loss_func(y_hat, y)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=True, logger=False, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        loss_epoch = self.trainer.callback_metrics['loss_epoch']
        self.logger.log_metrics({'loss_epoch':loss_epoch.item()}, step=self.trainer.current_epoch) # for diplay x-axis as epoch in mlflow.

    # --------------- validation loop ---------------
    def validation_step(self, batch, batch_idx):
        loss, dice = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_dice": dice} # , "val_loss": loss

        # self.log("val_dice", dice, on_step = False, on_epoch=True, prog_bar=True, logger=False, rank_zero_only=True)
        self.log("val_dice", dice, on_step = False, on_epoch=True, prog_bar=True, logger=False, sync_dist=True)
        return metrics

    def on_validation_end(self):
        val_dice = self.trainer.callback_metrics['val_dice']
        self.logger.log_metrics({'val_dice':val_dice.item()}, step=self.trainer.current_epoch)

    # --------------- test loop ---------------
    def test_step(self, batch, batch_idx):
        loss, dice = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_dice": dice}

        self.log("test_dice", dice, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return metrics

    def on_test_end(self):
        test_dice = self.trainer.callback_metrics['test_dice']
        self.logger.log_metrics({'test_dice':test_dice.item()}, step=self.trainer.current_epoch)

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']

        logits = self.model(x)  # probs = sigmoid(logits)
        y_hat = torch.sigmoid(logits)
        
        if isinstance(self.loss_func, nn.BCEWithLogitsLoss):
            loss = self.loss_func(logits, y)
        else:
            loss = self.loss_func(y_hat, y)
            
        dice = self.dice_coeff(y_hat, y)
        # acc = accuracy(y_hat, y)
        return loss, dice

    # --------------- optimizers ---------------
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams['learning_rate'])

    # --------------- not-neccessary ---------------
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        # items["nb_seen_classes"] = self.nb_seen_classes
        return items
