from torch.utils import data as data
from torchvision.transforms.functional import normalize

from data.data_util import (paired_paths_from_folder,
                            paired_paths_from_folder_debug,
                            paired_DP_paths_from_folder,
                            paired_paths_from_lmdb,
                            paired_paths_from_meta_info_file)
from data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
from data.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP

import random
import numpy as np
import torch
import cv2
import os

def absoluteFilePaths(directory, selected_file_list=None):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            if selected_file_list is None or selected_file_list in dirpath:
                yield os.path.abspath(os.path.join(dirpath, f)) 
                
def read_img(path):
        img = cv2.imread(path)
        img = img.astype(np.float32) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
class Dataset_Desnow(data.Dataset):
    def __init__(self, phase, dataroot_gt, dataroot_lq, geometric_augs, 
                 scale, gt_size=None) -> None:
        super().__init__()
        self.scale = scale
        self.gt_size = gt_size
        self.phase = phase
        
        self.gt_folder = dataroot_gt
        self.lq_folder = dataroot_lq
        
        self.gt_path = list(absoluteFilePaths(self.gt_folder))
        self.lq_path = list(absoluteFilePaths(self.lq_folder))
        # self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        if self.phase == 'train':
            self.geometric_augs = geometric_augs
    
    def __getitem__(self, index):
        scale = self.scale
        index = index % len(self.gt_path)
        gt_path = self.gt_path[index]
        lq_path = self.lq_path[index]

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        img_gt = read_img(gt_path)
        img_lq = read_img(lq_path)

        # augmentation for training
        if self.phase == 'train':
            gt_size = self.gt_size
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # flip, rotation
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                                        bgr2rgb=False,
                                        float32=True)
        else:            
            if 'gt_size' is not None:
                # padding
                img_gt, img_lq = padding(img_gt, img_lq, self.gt_size)

                # random crop
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, self.gt_size, scale,
                                                    gt_path)
                
            img_gt, img_lq = img2tensor([img_gt, img_lq],
                            bgr2rgb=False,
                            float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }
        
    def __len__(self):
        return len(self.gt_path)
    

class Dataset_Denoise(data.Dataset):
    def __init__(self, phase, dataroot_gt, dataroot_lq=None, geometric_augs=True, 
                 scale=1, gt_size=None, sigma_type=None, sigma_range=None,sigma_test=None) -> None:
        super().__init__()
        self.scale = scale
        self.gt_size = gt_size
        self.phase = phase
        
        self.gt_folder = dataroot_gt
        
        self.sigma_type = sigma_type
        self.sigma_range = sigma_range
        self.sigma_test = sigma_test
                
        self.gt_path = list(absoluteFilePaths(self.gt_folder))

        if self.phase == 'train':
            self.geometric_augs = geometric_augs
    
    def __getitem__(self, index):
        scale = self.scale
        index = index % len(self.gt_path)
        gt_path = self.gt_path[index]

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        img_gt = read_img(gt_path)
        img_lq = img_gt.copy()
        
        # augmentation for training
        if self.phase == 'train':
            gt_size = self.gt_size
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # flip, rotation
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                                        bgr2rgb=False,
                                        float32=True)
            
            if self.sigma_type == 'constant':
                sigma_value = self.sigma_range
            elif self.sigma_type == 'random':
                sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
            elif self.sigma_type == 'choice':
                sigma_value = random.choice(self.sigma_range)

            noise_level = torch.FloatTensor([sigma_value])/255.0
            noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise)
            
        else:            
            # HACK
            np.random.seed(seed=0)
            img_lq += np.random.normal(0, self.sigma_test/255.0, img_lq.shape)
            if self.gt_size is not None:
                # padding
                img_gt, img_lq = padding(img_gt, img_lq, self.gt_size)

                # random crop
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, self.gt_size, scale,
                                                    gt_path)
                
            img_gt, img_lq = img2tensor([img_gt, img_lq],
                            bgr2rgb=False,
                            float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }
        
    def __len__(self):
        return len(self.gt_path)
    
