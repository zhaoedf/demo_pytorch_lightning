

# ***************************************************************************
# import packages
# ***************************************************************************
import sys
import os
# import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import MLFlowLogger

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_bolts.callbacks import PrintTableMetricsCallback


# ***************************************************************************
# seed
# ***************************************************************************

seed_everything(42, workers=True)


# ***************************************************************************
# import self code
# ***************************************************************************

# --- data ---
from dataset import Caltech256
from data import ClsDataset, ClsDataModule

# --- model ---
import torchvision
model = torchvision.models.resnet18(pretrained=True)

# --- args ---
from args import args_trainer, args_model

# --- learner ---
from learner import ClsLearner


# ***************************************************************************
# import args
# ***************************************************************************
batch_size = args_model.batch_size
learning_rate = args_model.learning_rate

in_channel = args_model.in_channel
nb_classes = args_model.nb_classes
bilinear = args_model.bilinear
scale = args_model.scale

dataset_path = args_model.dataset_path
num_workers = args_model.num_workers
val_split_ratio = args_model.val_split_ratio


# ***************************************************************************
# dataset and datamodule
# ***************************************************************************

train_dataset = Caltech256(
    dataroot='/data/Public/Datasets/Caltech-256',
    train=True
)
test_dataset = Caltech256(
    dataroot='/data/Public/Datasets/Caltech-256',
    train=False
)

dm = ClsDataModule(
    train_dataset, test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    val_split_ratio = val_split_ratio
)


# ***************************************************************************
# model and learner
# ***************************************************************************


learner = ClsLearner(
    model=model,
    learning_rate=learning_rate
)


# ***************************************************************************
# trainer
# ***************************************************************************

# logger
mlflow_logger = MLFlowLogger(experiment_name="caltech", tracking_uri="http://localhost:10500")
run_id = mlflow_logger.run_id

# callbacks
print_table_metrics_callback = PrintTableMetricsCallback()

monitor_metric = 'val_acc'
mode = 'max'
early_stop_callback = EarlyStopping(
    monitor=monitor_metric,
    min_delta=0.00,
    patience=3,
    verbose=False,
    mode=mode,
    strict = True)

checkpoint_callback = ModelCheckpoint(
    dirpath=f'saved_models/{run_id}/',
    monitor=monitor_metric,
    filename='{epoch:02d}-{val_dice:.2f}',
    save_top_k=2,
    mode=mode,
    save_last=False
)



trainer = Trainer(
    accelerator=args_trainer.accelerator,
    gpus = args_trainer.gpus, # [0,1,7,8,9]  / -1
    # gpus = "1",
    max_epochs=args_trainer.max_epochs,
    progress_bar_refresh_rate=args_trainer.progress_bar_refresh_rate,
    check_val_every_n_epoch = args_trainer.check_val_every_n_epoch,
    weights_summary=args_trainer.weights_summary,
    callbacks = [early_stop_callback, checkpoint_callback],
    log_every_n_steps = args_trainer.log_every_n_steps, # default: 50
    logger = mlflow_logger,
    sync_batchnorm = args_trainer.sync_batchnorm,
    fast_dev_run = args_trainer.fast_dev_run
) # precision=16 [checked]



# ***************************************************************************
# training process
# ***************************************************************************

try:
    trainer.fit(learner, datamodule=dm)
    # test code [checked]
    dm.setup('test')
    trainer.test(learner, dataloaders=dm.test_dataloader())

    
except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)