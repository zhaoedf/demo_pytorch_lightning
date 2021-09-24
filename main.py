



from pytorch_lightning import Trainer, seed_everything

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from pl_bolts.callbacks import PrintTableMetricsCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger


# from continuum import ClassIncremental
# from continuum.tasks import split_train_val
# from continuum.datasets import MNIST, CIFAR10

from data.incremental_datamodule import IncrementalDataModule
from data.incremental_scenario import incremental_scenario

from learner.Base_learner import Base_learner

from args.args_trainer import args_trainer
from args.args_model import args_model
print(args_model)

# changhong code
from incremental_net.inc_net import IncrementalNet
inc_network = IncrementalNet("resnet32", pretrained=False, gradcam=False)


seed_everything(42, workers=True)

# model
learning_rate = args_model.learning_rate
learner = Base_learner(inc_network, learning_rate=learning_rate)


mlflow_logger = MLFlowLogger(experiment_name="test1", tracking_uri="http://localhost:10500")
run_id = mlflow_logger.run_id

# callbacks
# print_table_metrics_callback = PrintTableMetricsCallback()

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
    dirpath=f'saved_models/{run_id}/',
    monitor=monitor_metric,
    filename='sample-mnist-{epoch:02d}-{val_acc:.2f}',
    save_top_k=3,
    mode=mode,
    save_last=True
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




# PATH_DATASETS = "/data/Public/Datasets"
# increment=2
# initial_increment=2


inc_scenario = incremental_scenario(
    dataset_name = args_model.dataset,
    train_additional_transforms = [],
    test_additional_transforms = [],
    initial_increment = args_model.initial_increment,
    increment = args_model.increment,
    datasets_dir = args_model.datasets_dir
)
if trainer.is_global_zero:
    pass
inc_scenario.prepare_data()
train_scenario, test_scenario = inc_scenario.get_incremental_scenarios()
# print(inc_scenario.class_order)

nb_seen_classes = args_model.initial_increment

try:
    for task_id, taskset in enumerate(train_scenario):

        learner.model.update_fc(nb_seen_classes)
        dm = IncrementalDataModule(
            task_id = task_id, 
            train_taskset = taskset, 
            test_taskset = test_scenario[:task_id+1],
            dims = inc_scenario.dims, 
            nb_total_classes = inc_scenario.nb_total_classes,
            batch_size = args_model.batch_size,
            num_workers = args_model.num_workers,
            val_split_ratio = args_model.val_split_ratio)

        trainer.fit(learner, datamodule=dm)
        dm.setup('test')
        trainer.test(learner, dataloaders=dm.test_dataloader())

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