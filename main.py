

# ***************************************************************************
# import packages
# ***************************************************************************

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
from data import SegDataset, SegDataModule

# --- model ---
from model import UNet

# --- args ---
from args import args_trainer, args_model

# --- learner ---
from learner import SegLearner


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

train_dataset = SegDataset(
    imgs_dir=os.path.join(dataset_path, 'imgs'), 
    masks_dir=os.path.join(dataset_path, 'masks'), 
    scale=scale,
    mask_suffix='_mask'
)
test_dataset = SegDataset(
    imgs_dir=os.path.join(dataset_path, 'test_imgs'), 
    masks_dir=os.path.join(dataset_path, 'test_masks'), 
    scale=scale,
    mask_suffix='_mask'
)

dm = SegDataModule(
    train_dataset, test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    val_split_ratio = val_split_ratio
)


# ***************************************************************************
# model and learner
# ***************************************************************************

model = UNet(
    n_channels=in_channel,
    n_classes=nb_classes,
    bilinear=bilinear
)

learner = SegLearner(
    model=model,
    learning_rate=learning_rate
)


# ***************************************************************************
# trainer
# ***************************************************************************

# logger
mlflow_logger = MLFlowLogger(experiment_name="test1", tracking_uri="http://localhost:10500")
run_id = mlflow_logger.run_id

# callbacks
print_table_metrics_callback = PrintTableMetricsCallback()

monitor_metric = 'val_dice'
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