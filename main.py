from model import NeuralNetwork
from classifier import LitClassifier

from data import MNISTDataModule

from args_trainer import args

from pytorch_lightning import Trainer

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pl_bolts.callbacks import PrintTableMetricsCallback
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import MLFlowLogger


# model
learning_rate = 0.002
model = NeuralNetwork()
classifier = LitClassifier(model, learning_rate=learning_rate)

# data
dm = MNISTDataModule("/data/Public/Datasets")

# callbacks
print_table_metrics_callback = PrintTableMetricsCallback()

early_stop_callback = EarlyStopping(
    monitor="val_acc",
    min_delta=0.00,
    patience=3,
    verbose=False,
    mode="max",
    strict = True)

checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',
    filename='sample-mnist-{epoch:02d}-{val_acc:.2f}',
    save_top_k=3,
    mode='max',
    save_last=True
)

# Litmodel = LitMNIST("/data/Public/Datasets")
# print(hasattr(classfier, 'state'))

mlflow_logger = MLFlowLogger(experiment_name="test1", tracking_uri="http://localhost:10500")
# trainer = Trainer(
#     accelerator="ddp",
#     gpus = [8,9], # [0,1,7,8,9]  / -1
#     max_epochs=100,
#     progress_bar_refresh_rate=20,
#     check_val_every_n_epoch = 1,
#     weights_summary="full",
#     callbacks = [early_stop_callback, checkpoint_callback],
#     log_every_n_steps = 20, # default: 50
#     logger = mlflow_logger,
#     sync_batchnorm = True
# ) # precision=16 [checked]

trainer = Trainer(
    accelerator=args.accelerator,
    gpus = args.gpus, # [0,1,7,8,9]  / -1
    max_epochs=args.max_epochs,
    progress_bar_refresh_rate=args.progress_bar_refresh_rate,
    check_val_every_n_epoch = args.check_val_every_n_epoch,
    weights_summary=args.weights_summary,
    callbacks = [early_stop_callback, checkpoint_callback],
    log_every_n_steps = args.log_every_n_steps, # default: 50
    logger = mlflow_logger,
    sync_batchnorm = args.sync_batchnorm
) # precision=16 [checked]


# trainer = Trainer.from_argparse_args(args)
# trainer.logger = mlflow_logger
# trainer.callbacks = [early_stop_callback, checkpoint_callback]

# train
# https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#fit
trainer.fit(classifier,dm) # only lightningModule class can be used in function "fit"! so classifier is ok while model is not.

# test [checked]
# dm.setup('test')
# trainer.test(classifier, dataloaders=dm.test_dataloader())