# TODO
'''
1) prepare dataloader
2) refer: https://github.com/venkat8900/Robotic-Tool-Segmentation
'''

import os
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class RTS_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')

            #TODO
            # load the data and labels
            # Think opencv dataloading might work
            '''
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            '''
        else:
            vol_name = self.sample_list[idx].strip('\n')

            #TODO
            # load the data and labels
            # Think opencv dataloading might work
            '''
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]
            '''

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample



    
