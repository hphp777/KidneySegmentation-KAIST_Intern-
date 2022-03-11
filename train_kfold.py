######################################################
#                   nested kfold                     #
#       outer fold(total=3) : for the test           #
#           inner fold : for the training            #
#   Difference with the kfold : I did not reset the  #
#   model after inner fold to generalize the model.  #
#           Total training epoch = 18                #
#                                                    #
#                   By Haengbok Chung                #
######################################################

import argparse
from ast import Load
import logging
import sys
from pathlib import Path
# from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from tqdm import tqdm
import torchvision.transforms as transforms
import PIL.Image as PIL
import matplotlib.pyplot as plt

from utils.data_loading import BasicDataset, CarvanaDataset
from utils.kits_data import Loader
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet
from test import test
from sklearn.model_selection import ShuffleSplit
from train import FocalLoss, MixedLoss

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')
data_dir = "C:/Users/bispl2219/Desktop/Kidney_Segmentation/"
resize = 256
out_fold_num = 3
in_fold_num = 30

ss1 = ShuffleSplit(n_splits=out_fold_num, test_size=0.2, random_state=0)
ss2 = ShuffleSplit(n_splits=in_fold_num, test_size=0.25, random_state=0)

def nested_kfold_segmentation(
              epochs: int = 1,
              batch_size: int = 4,
              learning_rate: float = 0.001,
              save_checkpoint: bool = True,
              amp: bool = False):

    history = {'train_loss':[], 'val_score':[]}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)

    all_set = Loader(data_dir, flag ='kfold',  resize = resize)
    test_set = Loader(data_dir, flag='test', resize = resize)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)

    test_score = 0

    for fold, (train_val_idx, test_idx) in enumerate(ss1.split(all_set)):

        print('------------Outer fold no-{}-----------------------------------------------'.format(fold))
        
        # outer fold별로 모델을 각각 만듦: test데이터는 unseen data여야하기 때문에 연속적으로 훈련시키지 않음
        net = UNet(1, 2, bilinear=False)
        net = net.to(device)

        # weight = "C:/Users/bispl2219/Desktop/Kidney_Segmentation/checkpoints/checkpoint_fold" + str(fold+1) + ".pth"
        # net.load_state_dict(torch.load(weight, map_location=device))

        test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)
        test_loader = DataLoader(all_set, drop_last=True, **loader_args, sampler=test_subsampler)

        # inner fold에서는 모델을 연속성이 있게 훈련시킴. 최대한 다양한 상황에 노출시켜서 오버피팅을 막는것을 목표로 함
        for n_fold, (train_idx, val_idx) in enumerate(ss2.split(train_val_idx)):

            print('------------Inner fold no-{}---------------'.format(n_fold))

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
            train_loader = DataLoader(all_set, **loader_args, sampler=train_subsampler)
            val_loader = DataLoader(all_set, **loader_args, sampler=val_subsampler)

            # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
            optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
            # optimizerA = optim.Adam(net.parameters(), lr=0.01)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
            grad_scaler = torch.cuda.amp.GradScaler(enabled=amp) # Automatic Mixed Precision Training: speed up overall procedure
            # where some operations use the torch.float32 datatype and other operations use torch.float16
            criterion = nn.CrossEntropyLoss()
            global_step = 0
            mixedLoss = MixedLoss(10.0,2.0)

            fold_loss = 0
            fold_val_score = 0

            for epoch in range(epochs):

                net.train()
                epoch_loss = 0
                epoch_val_score = 0
                batch_len = 0
                val_len = 0

                with tqdm(total=len(train_idx), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                    for batch, (inputs, labels) in enumerate(train_loader):
                
                        batch_len += 1
                        images = inputs
                        true_masks = labels


                        assert images.shape[1] == net.n_channels, \
                            f'Network has been defined with {net.n_channels} input channels, ' \
                            f'but loaded images have {images.shape[1]} channels. Please check that ' \
                            'the images are loaded correctly.'

                        images = images.to(device=device, dtype=torch.float32)

                        true_masks = true_masks.to(device=device, dtype=torch.long)

                        with torch.cuda.amp.autocast(enabled=amp):
                            masks_pred = net(images)
                            true_masks = true_masks.squeeze(1)
                            #criterion(masks_pred, true_masks) 
                            mLoss = mixedLoss(F.softmax(masks_pred, dim=1).float(), F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float())
                            loss = criterion(masks_pred, true_masks) \
                                + dice_loss(F.softmax(masks_pred, dim=1).float(), # 1 - dice coefficient 
                                       F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)  \
                                + mLoss
                            

                        optimizer.zero_grad(set_to_none=True)
                        grad_scaler.scale(loss).backward()
                        grad_scaler.step(optimizer)
                        grad_scaler.update()

                        pbar.update(images.shape[0]) ##
                        global_step += 1
                        epoch_loss += loss.item()
                        pbar.set_postfix(**{'loss (batch)': loss.item()})

                        # Evaluation round
                        division_step = (len(train_idx) // (10 * batch_size))
                        if division_step > 0:
                            if global_step % division_step == 0:
                                val_score = evaluate(net, val_loader, device)
                                epoch_val_score += val_score
                                val_len += 1
                                scheduler.step(val_score)
            
                fold_loss += (epoch_loss/batch_len)
                fold_val_score += (epoch_val_score/val_len)

            history['train_loss'].append(fold_loss/epochs)
            history['val_score'].append(fold_val_score/epochs)


        # fold별로 결과 저장
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_fold{}.pth'.format(fold + 1)))

        result_save_dir = "C:/Users/bispl2219/Desktop/Kidney_Segmentation/checkpoints/results/"

        test_score += evaluate(net, test_loader, device)

        plt.plot(range(len(history['train_loss'])), history['train_loss'], label='Loss', color='red')
        plt.title('Train Loss history')
        plt.ylabel('loss')
        plt.xlabel('fold')
        plt.savefig(result_save_dir+'train_loss')

        plt.close('all')
        plt.clf()
        plt.cla()

        plt.plot(range(len(history['val_score'])), history['val_score'], label='Loss', color='blue')
        plt.title('Validation Score history')
        plt.ylabel('score')
        plt.xlabel('fold')
        plt.savefig(result_save_dir+'val_score')

        plt.close('all')
        plt.clf()
        plt.cla()

    print("Average Test Dice Coefficient: ", (test_score/out_fold_num).item())



if __name__ == "__main__":
    nested_kfold_segmentation()