
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from pytorch_lightning import LightningDataModule

from continuum import ClassIncremental
from continuum.tasks import split_train_val
from continuum.datasets import MNIST, CIFAR10

BATCH_SIZE = 64
PATH_DATASETS = "/data/Public/Datasets"

class CIFAR10IncrementalDataModule(LightningDataModule):

    def __init__(self, task_id, taskset, dims, num_classes):
        super().__init__()
        self.task_id = task_id
        self.taskset = taskset

        self.dims = dims
        self.num_classes = num_classes

    def prepare_data(self): # 这步其实就是为了download，即检查数据集是否存在而已。
        pass

    # def create_next_incremental_dataset(self):
    #     for task_id, taskset in enumerate(self.scenario):
    #         yield task_id, taskset

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            mnist_full = self.taskset
            self.mnist_train, self.mnist_val = split_train_val(mnist_full, val_split=0.1)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.mnist_test = dat


    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=BATCH_SIZE,num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=BATCH_SIZE,num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=BATCH_SIZE,num_workers=8)