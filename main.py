

import os

from pytorch_lightning import Trainer, seed_everything

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from pl_bolts.callbacks import PrintTableMetricsCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
# from pytorch_lightning.callbacks import RichProgressBar


from pytorch_lightning.loggers import MLFlowLogger



# from continuum import ClassIncremental
# from continuum.tasks import split_train_val
# from continuum.datasets import MNIST, CIFAR10

from data.incremental_datamodule import IncrementalDataModule
from data.incremental_scenario import incremental_scenario

from learner.Base_learner import Base_learner

from args.args_trainer import args_trainer
from args.args_model import args_model
# print(args_model)

# changhong code
from incremental_net.inc_net import IncrementalNet
inc_network = IncrementalNet("resnet32", pretrained=False, gradcam=False)

import sys

seed_everything(42, workers=True)

# model
learner = Base_learner(
    inc_network, 
    args_model
)

exp_name = "incremental_learning"
mlflow_logger = MLFlowLogger(experiment_name=exp_name, tracking_uri="http://localhost:10500")
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

# checkpoint_callback = ModelCheckpoint(
#     dirpath=f'saved_models/{run_id}/',
#     monitor=monitor_metric,
#     filename='sample-mnist-{epoch:02d}-{val_acc:.2f}',
#     save_top_k=3,
#     mode=mode,
#     save_last=True
# )
learning_rate_monitor_callback = LearningRateMonitor(
    logging_interval='epoch'
)

trainer = Trainer(
    accelerator=args_trainer.accelerator,
    gpus = args_trainer.gpus, # [0,1,7,8,9]  / -1
    # gpus = "1",
    max_epochs=args_trainer.max_epochs,
    progress_bar_refresh_rate=args_trainer.progress_bar_refresh_rate,
    check_val_every_n_epoch = args_trainer.check_val_every_n_epoch,
    weights_summary=args_trainer.weights_summary,
    callbacks = [learning_rate_monitor_callback], # early_stop_callback, checkpoint_callback
    log_every_n_steps = args_trainer.log_every_n_steps, # default: 50
    logger = mlflow_logger,
    sync_batchnorm = args_trainer.sync_batchnorm,
    fast_dev_run = args_trainer.fast_dev_run,
    num_sanity_val_steps = args_trainer.num_sanity_val_steps
) # precision=16 [checked]




# PATH_DATASETS = "/data/Public/Datasets"
# increment=2
# initial_increment=2

if trainer.is_global_zero:
    # print(mlflow_logger.run_id)
    print(f' \
            batch_size: {args_model.batch_size},\n \
            learning_rate: {args_model.learning_rate},\n \
            dataset: {args_model.dataset},\n \
            initial_increment: {args_model.initial_increment}, \n \
            increment: {args_model.increment} \
          ')
    
    ckpt_save_root_dir = 'saved_models/'
    ckpt_save_dir = os.path.join(ckpt_save_root_dir, run_id)
    if not os.path.exists(ckpt_save_dir):
        os.makedirs(ckpt_save_dir)
    
# mlflow_logger.log_hyperparams(args_model) [checked]


inc_scenario = incremental_scenario(
    dataset_name = args_model.dataset,
    train_additional_transforms = [],
    test_additional_transforms = [],
    initial_increment = args_model.initial_increment,
    increment = args_model.increment,
    datasets_dir = args_model.datasets_dir
)
# inc_scenario.setup()
train_scenario, test_scenario = inc_scenario.get_incremental_scenarios()
# print(inc_scenario.class_order)

nb_seen_classes = args_model.initial_increment


for task_id, taskset in enumerate(train_scenario):
    try:

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
        
    # [checked]
    # dm.setup('test')
    # trainer.test(learner, dataloaders=dm.test_dataloader())
    
        if trainer.is_global_zero: # control that only one device can print.
            print('*'*100)
            print(f'nb_seen_classes have been seen is: {nb_seen_classes}')
            
            trainer.save_checkpoint(os.path.join(ckpt_save_dir, f"{nb_seen_classes}-{learner.last_incremental_acc.item()}.ckpt"))

        learner.update_nb_seen_classes(nb_seen_classes)
        nb_seen_classes += args_model.increment

        del dm
        del trainer
        
        trainer = Trainer(
            accelerator=args_trainer.accelerator,
            gpus = args_trainer.gpus, # [0,1,7,8,9]  / -1
            # gpus = "1",
            max_epochs=args_trainer.max_epochs,
            progress_bar_refresh_rate=args_trainer.progress_bar_refresh_rate,
            check_val_every_n_epoch = args_trainer.check_val_every_n_epoch,
            weights_summary=args_trainer.weights_summary,
            callbacks = [learning_rate_monitor_callback], # early_stop_callback, checkpoint_callback RichProgressBar()
            log_every_n_steps = args_trainer.log_every_n_steps, # default: 50
            logger = mlflow_logger, # use the same logger, otherwise it will create new runs in mlflow.
            sync_batchnorm = args_trainer.sync_batchnorm,
            fast_dev_run = args_trainer.fast_dev_run,
            num_sanity_val_steps = args_trainer.num_sanity_val_steps
        ) # precision=16 [checked]
    except KeyboardInterrupt:
            try:
                break
                sys.exit(0)
            except SystemExit:
                os._exit(0)

# test [worked]
# dm.setup('test')
# trainer.test(classifier, dataloaders=dm.test_dataloader())
# test [not worked]
# trainer.test(classifier, datamodule=dm)

# 复现
'''
    查看iCaRL源代码，看看超参数设置
    查看pl中有关lr adjust的callback https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html 都看看
'''

#  提问，循环程序停止的问题，准备一个minimal example

#  发现trainer.fit 在训练第二个增量开始，只会训练一个epoch，这猜想是和trainer保持不变，存在状态记忆有关。