import os

import torch
import torch.nn.functional as F

from torch import nn

# Note - you must have torchvision installed for this example
from torchvision.datasets import CIFAR10, MNIST



# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# PATH_DATASETS = "/data/Public/Datasets"
# BATCH_SIZE = 32




# class LitMNIST(LightningModule):

#     def __init__(self, data_dir=PATH_DATASETS, hidden_size=64, learning_rate=2e-4):

#         super().__init__()

#         # We hardcode dataset specific stuff here.
#         self.data_dir = data_dir
#         self.num_classes = 10
#         self.dims = (1, 28, 28)
#         channels, width, height = self.dims
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307, ), (0.3081, )),
#         ])

#         self.hidden_size = hidden_size
#         self.learning_rate = learning_rate

#         # Build model
#         self.model = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(channels * width * height, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_size, self.num_classes),
#         )

#     def forward(self, x):
#         x = self.model(x)
#         return F.log_softmax(x, dim=1)

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = F.nll_loss(logits, y)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = F.nll_loss(logits, y)
#         preds = torch.argmax(logits, dim=1)
#         acc = accuracy(preds, y)
#         self.log('val_loss', loss, prog_bar=True)
#         self.log('val_acc', acc, prog_bar=True)
#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
#         return optimizer

#     ####################
#     # DATA RELATED HOOKS
#     ####################

#     def prepare_data(self):
#         # download
#         MNIST(self.data_dir, train=True, download=True)
#         MNIST(self.data_dir, train=False, download=True)

#     def setup(self, stage=None):

#         # Assign train/val datasets for use in dataloaders
#         if stage == 'fit' or stage is None:
#             mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
#             self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

#         # Assign test dataset for use in dataloader(s)
#         if stage == 'test' or stage is None:
#             self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

#     def train_dataloader(self):
#         return DataLoader(self.mnist_train, batch_size=BATCH_SIZE)

#     def val_dataloader(self):
#         return DataLoader(self.mnist_val, batch_size=BATCH_SIZE)

#     def test_dataloader(self):
#         return DataLoader(self.mnist_test, batch_size=BATCH_SIZE)