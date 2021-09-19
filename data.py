
import numpy as np
import torch
import logging
import os

from os.path import splitext
from os import listdir

from glob import glob

from PIL import Image

from torch.utils.data import Dataset, random_split


class SegDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix='_mask'):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        # print(self.ids)
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(os.path.join(self.masks_dir, idx+self.mask_suffix+'.*'))  # glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(os.path.join(self.imgs_dir, idx+'.*'))
        
        '''
            mask_file is like: .../idx_mask.png
            img_file is like: .../idx.png
        '''
        #print(len(img_file),len(mask_file),self.masks_dir + idx + self.mask_suffix + '.*')
        '''
            old problem, cos here uses "string add" instead of os.path.join,
            so the variables[mask_file, img_file] above, you will have to pay attention to the "/" in the last of the self.[masks_dir, imgs_dir].
        '''
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        # mask = Image.open(mask_file[0]) # old
        mask = Image.open(mask_file[0]).convert('L') # new
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)
        #print(img.shape,mask.shape)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }




from torchvision.transforms import ToTensor, Lambda, Compose
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from pytorch_lightning import LightningDataModule


BATCH_SIZE = 16

class SegDataModule(LightningDataModule):

    def __init__(self, 
                 train_dataset, 
                 test_dataset,
                 batch_size:int,
                 num_workers:int,
                 val_split_ratio:float):
        super().__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split_ratio = val_split_ratio

    def prepare_data(self): # 这步其实就是为了download，即检查数据集是否存在而已。
        pass


    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            full = self.train_dataset
            
            n_train = int(len(full)*self.val_split_ratio)
            n_val = len(full) - n_train
            
            self.train, self.val = train, val = random_split(full, [n_train, n_val])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test = self.test_dataset


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size,num_workers=self.num_workers)

    def val_dataloader(self):
        if self.val_split_ratio == 0.0:
            return None # return None will force trainer to disable Validation loop.
        else:
            return DataLoader(self.val, batch_size=self.batch_size,num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size,num_workers=self.num_workers)