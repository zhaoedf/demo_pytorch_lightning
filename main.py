from model import NeuralNetwork
from classifier import LitClassifier

from data import MNISTDataModule

from args_trainer import args

from pytorch_lightning import Trainer, seed_everything

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pl_bolts.callbacks import PrintTableMetricsCallback
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import MLFlowLogger

from torchvision import transforms

# from continuum import ClassIncremental
# from continuum.tasks import split_train_val
# from continuum.datasets import MNIST, CIFAR10

from incremental_data import IncrementalDataModule
from incremental_scenario import incremental_scenario

# changhong code
from utils.inc_net import IncrementalNet
inc_network = IncrementalNet("resnet32", False)


seed_everything(42, workers=True)

# model
learning_rate = 0.002
# model = NeuralNetwork()
# classifier = LitClassifier(model, learning_rate=learning_rate)
# TODO
# 封装，把网络生成写在Classifier里面，并且改个名字
classifier = LitClassifier(inc_network, learning_rate=learning_rate)

# callbacks
print_table_metrics_callback = PrintTableMetricsCallback()

monitor_metric = 'loss_epoch'
mode = 'min'
early_stop_callback = EarlyStopping(
    monitor=monitor_metric,
    min_delta=0.00,
    patience=3,
    verbose=False,
    mode=mode,
    strict = True)

checkpoint_callback = ModelCheckpoint(
    monitor=monitor_metric,
    filename='sample-mnist-{epoch:02d}-{val_acc:.2f}',
    save_top_k=3,
    mode=mode,
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
    # gpus = "1",
    max_epochs=args.max_epochs,
    progress_bar_refresh_rate=args.progress_bar_refresh_rate,
    check_val_every_n_epoch = args.check_val_every_n_epoch,
    weights_summary=args.weights_summary,
    callbacks = [early_stop_callback, checkpoint_callback],
    log_every_n_steps = args.log_every_n_steps, # default: 50
    logger = mlflow_logger,
    sync_batchnorm = args.sync_batchnorm,
    fast_dev_run = args.fast_dev_run
) # precision=16 [checked]


# trainer = Trainer.from_argparse_args(args)
# trainer.logger = mlflow_logger
# trainer.callbacks = [early_stop_callback, checkpoint_callback]

# train
# https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#fit
# print(dir(dm))

PATH_DATASETS = "/data/Public/Datasets"
increment=2
initial_increment=2
# transform = [
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307, ), (0.3081, )),
#         ]

# scenario = ClassIncremental(
#         CIFAR10(PATH_DATASETS, train=True),
#         increment=increment,
#         initial_increment=initial_increment,
#         transformations=transform
#         )

# TODO
# 后续可以把incrementtal_scenario作为基类，然后每个数据集写成一个XX_incrementtal_scenario。
# 这样可以更加独立和方便阅读
inc_scenario = incremental_scenario(
    dataset_name = 'CIFAR10',
    train_additional_transforms = [],
    test_additional_transforms = [],
    initial_increment = initial_increment,
    increment = increment
)
if trainer.is_global_zero:
    pass
inc_scenario.prepare_data()
train_scenario, test_scenario = inc_scenario.get_incremental_scenarios()
# print(inc_scenario.class_order)

nb_seen_classes = initial_increment
val_split_ratio = 0.0

try:
    for task_id, taskset in enumerate(train_scenario):

        classifier.model.update_fc(nb_seen_classes)
        dm = IncrementalDataModule(
            task_id = task_id, 
            train_taskset = taskset, 
            test_taskset = test_scenario[:task_id+1],
            dims = inc_scenario.dims, 
            nb_total_classes = inc_scenario.nb_total_classes,
            batch_size = 64,
            num_workers = 2,
            val_split_ratio = val_split_ratio)

        # trainer.fit(classifier, datamodule=dm) # only lightningModule class can be used in function "fit"! so classifier is ok while model is not.
        # dm.setup('fit')
        #trainer.fit(classifier, train_dataloaders = dm.train_dataloader())
        trainer.fit(classifier, datamodule=dm)
        dm.setup('test')
        trainer.test(classifier, dataloaders=dm.test_dataloader())
        # trainer.test(classifier, datamodule=dm)

        if trainer.is_global_zero: # control that only one device can print.
            print('*'*100)
            print(f'nb_seen_classes have been seen is: {nb_seen_classes}')

        classifier.set_nb_seen_classes(nb_seen_classes)
        nb_seen_classes += increment

        del dm
except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

# test [worked]
# dm.setup('test')
# trainer.test(classifier, dataloaders=dm.test_dataloader())
# test [not worked]
# trainer.test(classifier, datamodule=dm)