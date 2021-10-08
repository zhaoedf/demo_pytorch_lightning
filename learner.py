import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torch.nn as nn 

from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy


class ClsLearner(pl.LightningModule):

    # --------------- computations ---------------
    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore='model')
        
        self.loss_func = nn.BCEWithLogitsLoss()


    # --------------- training loop ---------------
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        
        # if isinstance(self.loss_func, nn.BCEWithLogitsLoss):
        #     loss = self.loss_func(logits, y)
        # else:
        #     loss = self.loss_func(y_hat, y)

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
        loss, val_acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": val_acc} # , "val_loss": loss

        # self.log("val_dice", dice, on_step = False, on_epoch=True, prog_bar=True, logger=False, rank_zero_only=True)
        self.log("val_acc", val_acc, on_step = False, on_epoch=True, prog_bar=True, logger=False, sync_dist=True)
        return metrics

    def on_validation_end(self):
        val_acc = self.trainer.callback_metrics['val_acc']
        self.logger.log_metrics({'val_acc':val_acc.item()}, step=self.trainer.current_epoch)

    # --------------- test loop ---------------
    def test_step(self, batch, batch_idx):
        loss, test_acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": test_acc}

        self.log("test_acc", test_acc, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return metrics

    def on_test_end(self):
        test_acc = self.trainer.callback_metrics['test_acc']
        self.logger.log_metrics({'test_acc':test_acc.item()}, step=self.trainer.current_epoch)

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        return loss, acc

    # --------------- optimizers ---------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams['learning_rate'])
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=0.1,
            patience=2,
            cooldown=3,
            min_lr=1e-6
        )
        return{
            "optimizer": optimizer ,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_acc",
            }
        }

        # return [optimizer], [lr_scheduler]

    # --------------- not-neccessary ---------------
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        # items["nb_seen_classes"] = self.nb_seen_classes
        return items
