import os
import torch
from torch.utils.data import Dataset
import cv2
from pathlib import Path


class RTS_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, task, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.task = task

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        task_dict = {'binary': 'binary_masks', 'instrument': 'instruments_masks', 'parts': 'parts_masks'}
        
        slice_path = self.sample_list[idx].strip('\n')
        base_path = Path(slice_path).resolve().parents[1]
        img_name = os.path.basename(slice_path)
        label_path = os.path.join(base_path, task_dict[self.task], img_name)
        image = cv2.imread(slice_path)
        label = cv2.imread(label_path)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample



    
