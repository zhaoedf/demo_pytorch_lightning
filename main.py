from model import NeuralNetwork
from classifier import LitClassifier

from data import MNISTDataModule

from args_trainer import args

from pytorch_lightning import Trainer

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pl_bolts.callbacks import PrintTableMetricsCallback
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import MLFlowLogger

from torchvision import transforms

from continuum import ClassIncremental
from continuum.tasks import split_train_val
from continuum.datasets import MNIST, CIFAR10

from incremental_data import CIFAR10IncrementalDataModule

# changhong code
from utils.inc_net import IncrementalNet
inc_network = IncrementalNet("resnet32", False)

# model
learning_rate = 0.002
model = NeuralNetwork()
# classifier = LitClassifier(model, learning_rate=learning_rate)
classifier = LitClassifier(inc_network, learning_rate=learning_rate)

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
    # gpus = "1",
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
# print(dir(dm))

PATH_DATASETS = "/data/Public/Datasets"
increment=2
initial_increment=2
transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, )),
        ]
# TODO scenario 做成某个类返回的结果。类的init参数为increment相关设置。 类中可以根据不同的类返回不同的数据集。
scenario = ClassIncremental(
        CIFAR10(PATH_DATASETS, train=True),
        increment=increment,
        initial_increment=initial_increment,
        transformations=transform
        )

nb_classes = initial_increment
for task_id, taskset in enumerate(scenario):
    # data
    # TODO 这个dims, num_classes的信息通过上一个TODO的类提供。

    classifier.model.update_fc(nb_classes)
    dm = CIFAR10IncrementalDataModule(task_id, taskset, (1,28,28), 10)

    trainer.fit(classifier,dm) # only lightningModule class can be used in function "fit"! so classifier is ok while model is not.

    if trainer.is_global_zero:
        print('*'*100)
        print(f'nb_classes have been seen is: {nb_classes}')
    nb_classes += increment


# test [checked]
# dm.setup('test')
# trainer.test(classifier, dataloaders=dm.test_dataloader())