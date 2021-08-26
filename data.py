
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from pytorch_lightning import LightningDataModule



# training_data = datasets.MNIST(
#     root="/data/Public/Datasets",
#     train=True,
#     download=False,
#     transform=ToTensor(),
# )

# # Download test data from open datasets.
# test_data = datasets.MNIST(
#     root="/data/Public/Datasets",
#     train=False,
#     download=True,
#     transform=ToTensor(),
# )

# batch_size = 64

# # Create data loaders.
# train_dataloader = DataLoader(training_data, batch_size=batch_size)
# val_dataloader = DataLoader(test_data, batch_size=batch_size)


BATCH_SIZE = 64
PATH_DATASETS = "/data/Public/Datasets"

class MNISTDataModule(LightningDataModule):

    def __init__(self, data_dir: str = PATH_DATASETS):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, )),
        ])

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        # download
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.mnist_test = datasets.MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=BATCH_SIZE,num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=BATCH_SIZE,num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=BATCH_SIZE,num_workers=8)