import pytorch_lightning as pl
import torch.nn.functional as F
import torch


from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy

class Base_learner(pl.LightningModule):

    # --------------- computations ---------------
    def __init__(self, model, args_model):
        super().__init__()
        self.model = model
        self.save_hyperparameters(args_model, ignore='model')

        self.nb_seen_classes = 0
        self.current_increment_index = self.nb_seen_classes
        self.register_buffer("last_incremental_acc", torch.zeros(1), persistent=False) # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.flag = 0


    # --------------- training loop ---------------
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        x, y, t = batch
        y_hat = self.model(x)['logits']
        loss = F.cross_entropy(y_hat, y)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log(f"{self.current_increment_index}_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log(f"{self.current_increment_index}_loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=True, logger=False, sync_dist=True)
        return loss


    def on_train_epoch_end(self):
        loss_epoch = self.trainer.callback_metrics[f"{self.current_increment_index}_loss_epoch"] # loss_epoch.shape: torch.Size([]), i.e. 1d tensor, corresponding to "epoch"!
        self.logger.log_metrics({f"{self.current_increment_index}_loss_epoch":loss_epoch.item()}, step=self.trainer.current_epoch)


    # --------------- validation loop ---------------
    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        '''
            the acc is computed on the test dataset which contains all *seen* classes test samples.
            e.g. when nb_seen_classes = 4 and the dataset is CIFAR10(ini_incre=2, incre=2), then \
                the current test dataset will contain *6(4{seen}+2{incre}) classes' test samples* .
        '''
        metrics = {"incremental_test_acc": acc} # here, due the feature of pl, we use val_loop to perform *test* actually
        self.log("incremental_test_acc", acc, on_step = False, on_epoch=True, prog_bar=True, logger=False, sync_dist=True)
        return metrics


    def on_validation_end(self):
        '''
            *the code below has nothing to do with the sync_dist*.
            it existence is only for the reason that mlflow originally do not support x-axis as epoch and \
            the code below will enable using epoch as x-axis.
            [on_test_end, on_train_epoch_end is for the same reason.]
        ''' 
        incremental_test_acc = self.trainer.callback_metrics['incremental_test_acc']
        self.last_incremental_acc = incremental_test_acc
        self.logger.log_metrics({'incremental_test_acc': incremental_test_acc.item()}, step = self.current_increment_index) # self.trainer.current_epoch


    # --------------- test loop ---------------
    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        self.log("test_acc", acc, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return acc


    def on_test_end(self):
        test_acc = self.trainer.callback_metrics['test_acc']
        self.logger.log_metrics({'test_acc':test_acc.item()}, step=self.trainer.current_epoch)


    def _shared_eval_step(self, batch, batch_idx):
        x, y, t = batch
        y_hat = self.model(x)['logits']
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        return loss, acc

    # --------------- optimizers ---------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams['learning_rate'])
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[49,63], gamma=0.2)
        
        return [optimizer], [lr_scheduler]

    # --------------- not-neccessary ---------------
    # In the case where you want to scale your inference, you should be using predict_step()
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        # demo
        x, y = batch
        return self.model(x)['logits']


    def update_nb_seen_classes(self, new_nb_seen_classes):
        self.nb_seen_classes = new_nb_seen_classes
        self.current_increment_index = (self.nb_seen_classes - self.hparams['initial_increment']) // self.hparams['increment'] + 1


    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items["nb_seen_classes"] = self.nb_seen_classes
        
        items[f"[:{(self.current_increment_index-1)*self.hparams['increment']+self.hparams['initial_increment']}]_acc"] = self.last_incremental_acc.item()
        return items


