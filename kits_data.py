from ctypes import resize
import os
from scipy import io
import numpy as np
import sys
from torch.utils.data import Dataset
import torch
import random
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
from numpy import newaxis
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import PIL.Image as PIL
# import albumentations as A

class Loader(Dataset): # custom dataset

    def KITSdataloader(self, index): # index = number

        scaler = MinMaxScaler()

        # Load Image
        # image, mask size가 동일한지 모르겠다: 동일
        # 근데 잘린 image의 사이즈가 다름. resize 필요
        # n_slice=random.randint(0, 511) # 

        resize = 256
        
        kidney = io.loadmat(self.dir + "data/img" + str(self.imgnames[index]) + ".mat")
        kidney = kidney['data']
        mask = io.loadmat(self.dir + "data/mask" + str(self.imgnames[index]) + ".mat")
        mask = mask['data']                    


        # kidney = kidney / kidney.max()
        # kidney += 1

        mean = np.mean(kidney)
        std = np.std(kidney)

        # print(kidney)

        transform1 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([resize,resize], PIL.NEAREST),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize(mean, std, inplace=False),
        ])
        

        transform2 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([resize,resize], PIL.NEAREST),
        transforms.ToTensor(),
        ])

        # transform = A.Compose([
        # A.HorizontalFlip(p=0.5),
        # A.RandomBrightnessContrast(p=0.2),
        # ])

        kidney = transform1(kidney)
        mask = transform2(mask)

        # print(kidney)

        # fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
        # ax1.imshow(kidney[0])
        # ax1.set_title('input')
        # ax2.imshow(mask[0])
        # ax2.set_title('mask')
        # plt.show()

        return kidney, mask

    def __init__(self, dir, flag, resize):
        self.dir = dir
        self.resize = resize
        self.flag = flag

        # Split Test and Train

        if self.flag == 'train':
            # self.imgnames = self.lines[:50] # Tip : you can adjust the number of images and run the quickly during debugging.
            # 12822
            self.imgnames = range(12822)

        elif self.flag == 'val':
            # self.imgnames = self.lines[50:60] # Tip : you can adjust the number of images and run the quickly during debugging.
            # 14424
            self.imgnames = range(12822,14424)
        elif self.flag == 'kfold':
            self.imgnames = range(16029)
        else:
            #16029
            self.imgnames = range(14424,16029)

        self.cls = {0: 0, 255: 1}  # edge # grayscale

    def __len__(self):
        return len(self.imgnames)

    def __getitem__(self, index): # 여기에 들어가는 index는?
        if torch.is_tensor(index):
            index = index.tolist()
        images, masks= self.KITSdataloader(index)

        return images, masks