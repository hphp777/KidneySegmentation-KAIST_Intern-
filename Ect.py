from ctypes import resize
import gc
from scipy.io import savemat
from scipy import io

from tqdm import tqdm
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

import numpy as np
import PIL.Image as PIL
from PIL import Image
import nibabel as nib
import glob
import matplotlib.pyplot as plt
from tifffile import imsave
import random
from sklearn.preprocessing import MinMaxScaler

import UNetImplementation as UNet
from datasets import Loader
from modules import *

scaler = MinMaxScaler()

def sizeCheck():

    n_slice=300
    # random.randint(0, 511)

    mask = nib.load('C:/Users/bispl2219/Desktop/Kidney_Segmentation/data_mask/case_00000/segmentation.nii.gz').get_data() 
    mask=scaler.fit_transform(mask.reshape(-1, mask.shape[-1])).reshape(mask.shape) # numpy array
    kidney = nib.load('C:/Users/bispl2219/Desktop/Kidney_Segmentation/data_raw/case_00000/imaging.nii.gz').get_data() 
    kidney=scaler.fit_transform(kidney.reshape(-1, kidney.shape[-1])).reshape(kidney.shape) # numpy array

    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
    ax1.imshow(mask[mask.shape[0]//2])
    ax1.set_title('Mask')
    ax2.imshow(kidney[kidney.shape[0]//2])
    ax2.set_title('Image')
    plt.show()
    # ax1.imshow(mask[48])


    # mask = mask[:,:,n_slice]
    # kidney = kidney[:,:,n_slice]
    # pil_image=Image.fromarray(kidney)
    # pil_image.show()

    print("mask: ", mask.shape,", kidney:",kidney.shape) # (z,x,y)

def segment():

    cls_invert = {0: 0, 1:255}

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = UNet.Unet(1, 2) # parameter: (input_channel, output_channel) -> (gray_scale, kidneyOrNot)
    model = model.to(device)

    kidney =  nib.load("C:/Users/bispl2219/Desktop/Kidney_Segmentation/data_raw/case_00000/imaging.nii.gz").get_fdata() 
    kidney=scaler.fit_transform(kidney.reshape(-1, kidney.shape[-1])).reshape(kidney.shape) # numpy array
    kidney = kidney[kidney.shape[0]//2].astype(np.float32)

    outputs = model(kidney)
    outputs = outputs.squeeze(1).cpu()

    preds = torch.argmax(outputs, dim=3).float()

    for j in range(preds.shape[0]):
        for k in range(preds.shape[1]):
            temp_gray = np.zeros((preds.shape[0], preds.shape[1]))
            temp_gray[j][k] = cls_invert[preds[j][k]]

    fig, (ax1) = plt.subplots(1,1, figsize = (12, 6))
    ax1.imshow(preds)
    ax1.set_title('prediction')
    plt.show()



def emptyCUDA():
    gc.collect()
    torch.cuda.empty_cache()


def FilesPreprocessing():

    n = 0

    for index in range(210):

        dir = "C:/Users/bispl2219/Desktop/Kidney_Segmentation/"

        # slice를 여러개 잘라서 신장이 있는 slice만 저장하기

        # 3D파일 불러오기
        mask = nib.load(dir + "data_mask_old/case_" + str(format(index, '05')) + "/segmentation.nii.gz").get_fdata()
        kidney = nib.load(dir + "data_raw_old/case_" + str(format(index, '05')) +"/imaging.nii.gz").get_fdata() 

        for s in range(mask.shape[0]):

            temp_mask = mask[s].astype(np.float32)
            temp_kidney = kidney[s].astype(np.float32)

            # 마스크 가공
            for i in range(temp_mask.shape[0]):
                for j in range(temp_mask.shape[1]):
                    if temp_mask[i][j] == 2:
                        temp_mask[i][j] = 1
                    elif temp_mask[i][j] == 0.5:
                        temp_mask[i][j] = 1

            
            # 신장이 있는 경우에 파일 저장하기
            save_dir = "C:/Users/bispl2219/Desktop/Kidney_Segmentation/data_all/mask" + str(n) + ".mat"
            savemat(save_dir, {'data': temp_mask})
            save_dir = "C:/Users/bispl2219/Desktop/Kidney_Segmentation/data_all/img" + str(n) + ".mat"
            savemat(save_dir, {'data': temp_kidney}) 

            n += 1

        print(index , " / 210 progressed...")


def clearValue():
    
    # length = 16028
    dir = "C:/Users/bispl2219/Desktop/Kidney_Segmentation/"
    n=0
    for index in range(210):
        mask = nib.load(dir + "data_mask_old/case_" + str(format(index, '05')) + "/segmentation.nii.gz").get_fdata()
        kidney = nib.load(dir + "data_raw_old/case_" + str(format(index, '05')) +"/imaging.nii.gz").get_fdata() 

        for s in range(mask.shape[0]):
            
            flag = False
            temp_mask = mask[s].astype(np.float32)
            temp_kidney = kidney[s].astype(np.float32)

            for y in range(temp_mask.shape[0]):
                for x in range(temp_mask.shape[1]):
                    if temp_mask[y][x] == 1:
                        flag = True
                        break
                if flag == True:
                    break

            if flag == False: # If there is no kidney, pass
                continue

            # for i in range(temp_mask.shape[0]):
            #     for j in range(temp_mask.shape[1]):
            #         if temp_mask[i][j] == 1: # kidney -> 0
            #             temp_mask[i][j] = 0
            #         elif temp_mask[i][j] == 2: # tumor -> ok
            #             temp_mask[i][j] = 1
            #         elif temp_mask[i][j] == 0.5: # unvalid value
            #             temp_mask[i][j] = 0

            save_dir = "C:/Users/bispl2219/Desktop/Kidney_Segmentation/data/kidneyTumor" + str(n) + ".mat"
            savemat(save_dir, {'data': temp_kidney})
            # save_dir = "C:/Users/bispl2219/Desktop/Kidney_Segmentation/data/mask" + str(n) + ".mat"
            # savemat(save_dir, {'data': temp_mask})

            n += 1

        print(index ,"/ 209 progressed...")

def showOriginal():
    dir = "C:/Users/bispl2219/Desktop/Kidney_Segmentation/"

    mask = io.loadmat(dir + "data/maskTumor100.mat")
    mask = mask['data']
    kidney = io.loadmat(dir + "data/kidneyTumor100.mat")
    kidney = kidney['data']

    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
    ax1.imshow(kidney)
    ax1.set_title('input')
    ax2.imshow(mask)
    ax2.set_title('mask')
    plt.savefig(dir + "checkpoints/results/original")

def saveCheck():

    dir = "C:/Users/bispl2219/Desktop/Kidney_Segmentation/"

    kidney = nib.load(dir + "data_raw_old/case_00001/imaging.nii.gz").get_fdata() 
    # kidney=scaler.fit_transform(kidney.reshape(-1, kidney.shape[-1])).reshape(kidney.shape) # numpy array

    kidneyOriginal = kidney[50].astype(np.float32)

    fig, (ax1) = plt.subplots(1,1, figsize = (12, 6))
    ax1.imshow(kidneyOriginal)
    ax1.set_title('original')
    plt.show()

    save_dir = "C:/Users/bispl2219/Desktop/Kidney_Segmentation/ori.mat"
    savemat(save_dir, {'data': kidneyOriginal})

    dir = "C:/Users/bispl2219/Desktop/Kidney_Segmentation/"
    kidneyMat = io.loadmat(dir + "ori.mat")
    kidneyMat = kidneyMat['data']

    for y in range(kidneyMat.shape[0]):
        for x in range(kidneyMat.shape[1]):
            if kidneyOriginal[y][x] != kidneyMat[y][x]:
                print("Value Changed!")
                return

    print("Value Unchanged!")

def sample() :
    dir = "C:/Users/bispl2219/Desktop/Kidney_Segmentation/"

    kidney = io.loadmat(dir + "data/imgClip1.mat")
    kidney = kidney['data']

    fig, (ax1) = plt.subplots(1,1, figsize = (12, 6))
    ax1.imshow(kidney)
    ax1.set_title('original')
    plt.show()

def Segment():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_dir = "C:/Users/bispl2219/Desktop/Kidney_Segmentation/"
    resize = 256

    kidney = io.loadmat(data_dir + "data/img3800.mat")
    kidney = kidney['data']

    transform1 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([resize,resize], PIL.NEAREST),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std, inplace=False),
    ])
    kidney = transform1(kidney)
    kidney = kidney.to(device)


    model = UNet.Unet(1, 2) # parameter: (input_channel, output_channel) -> (gray_scale, kidneyOrNot)
    
    PATH = 'C:/Users/bispl2219/Desktop/Kidney_Segmentation/output/model/16.pth'
    # PATH = './trained_model/resnet_encoder_unet.pth'
    checkpoint = torch.load(PATH, map_location=device)
    model.load_state_dict(checkpoint)

    model = model.to(device)

    output = model(kidney)

    print(output)

def clipping():

    # Image value clipping
    for n in tqdm(range(16029)):

        dir1 = "C:/Users/bispl2219/Desktop/Kidney_Segmentation/data/img" + str(n) + ".mat"
        dir2 = "C:/Users/bispl2219/Desktop/Kidney_Segmentation/data/mask" + str(n) + ".mat"

        kidney = io.loadmat(dir1)
        kidney = kidney['data']
        mask = io.loadmat(dir2)
        mask = mask['data']

        min_value = 255
        max_value = 0

        # get the range of value
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if mask[y][x] == 1:
                    if kidney[y][x] > max_value:
                        max_value = kidney[y][x]
                    if kidney[y][x] < min_value:
                        min_value = kidney[y][x]

        # replace
        for y in range(kidney.shape[0]):
            for x in range(kidney.shape[1]):
                if kidney[y][x] < min_value:
                    kidney[y][x] = min_value
                if kidney[y][x] > max_value:
                    kidney[y][x] = max_value

        # save
        save_dir = "C:/Users/bispl2219/Desktop/Kidney_Segmentation/data/imgClip" + str(n) + ".mat"
        savemat(save_dir, {'data': kidney})

FilesPreprocessing()

