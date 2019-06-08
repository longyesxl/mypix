import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
import random
from pathlib import Path
class myDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset_path,val_set=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.save_dir = Path(dataset_path)
        self.save_dir.mkdir(exist_ok=True)
        self.img_dir = self.save_dir.joinpath('images')
        self.img_dir.mkdir(exist_ok=True)
        self.train_dir = self.save_dir.joinpath('train')
        self.train_dir.mkdir(exist_ok=True)
        self.train_img_dir = self.train_dir.joinpath('train_img')
        self.train_img_dir.mkdir(exist_ok=True)
        self.train_label_dir = self.train_dir.joinpath('train_label')
        self.train_label_dir.mkdir(exist_ok=True)

        self.train2_dir = self.save_dir.joinpath('train2')
        self.train2_dir.mkdir(exist_ok=True)
        self.train2_img_dir = self.train2_dir.joinpath('train2_img')
        self.train2_img_dir.mkdir(exist_ok=True)
        self.train2_label_dir = self.train2_dir.joinpath('train2_label')
        self.train2_label_dir.mkdir(exist_ok=True)
        self.val_set=val_set
        self.dataset_path=dataset_path
        self.dataset_name_list=os.listdir(dataset_path)

    def __len__(self):
        if self.val_set:
            return 1
        else:
            return 1212

    def __getitem__(self, idx):
        if self.val_set:
            indexs=39
            imageout=cv2.imread(str(self.train_img_dir.joinpath(f'img_{indexs:04d}.png'))).astype(np.float32)/255.0
            imagein=cv2.imread(str(self.train_label_dir.joinpath(f'label_{indexs:04d}.png'))).astype(np.float32)/255.0
            img_out=torch.from_numpy(imageout.transpose((2,0,1)))
            img_in=torch.from_numpy(imagein.transpose((2,0,1)))
            sample = {'img_in': img_in, 'img_out': img_out}
            return sample
        else:
            imageout=cv2.imread(str(self.train_img_dir.joinpath(f'img_{idx:04d}.png'))).astype(np.float32)/255.0
            imagein=cv2.imread(str(self.train_label_dir.joinpath(f'label_{idx:04d}.png'))).astype(np.float32)/255.0
            img_out=torch.from_numpy(imageout.transpose((2,0,1)))
            img_in=torch.from_numpy(imagein.transpose((2,0,1)))
            sample = {'img_in': img_in, 'img_out': img_out}
            return sample
