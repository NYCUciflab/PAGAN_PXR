import os
import time
import torch.nn as nn
import pandas as pd
import numpy as np
from shutil import copyfile

from config import *
from setting import *
from utils.dataset import build_dataset, build_CropDataset
from utils.model import Discriminator, PAGenerator
from utils.training import train_by_epoch
from utils.loss import BCEloss


# information
struct_time   = time.localtime()
year_month    = time.strftime("%Y-%m%2d", struct_time)
file_name     = year_month+'-'+model_name+'-'+GAN_loss+'-'+annotaton+'-'+str(image_size)
result_dir    = f'./results/{file_name}/'
weight_dir    = f'./weights/{file_name}/'
saved_img_dir = result_dir+'saved_img/'


# mkdir
try:
    os.mkdir(result_dir)
    print('result DIR CREATED')
except:
    print("result DIR ALREADY EXIST")
try:
    os.mkdir(weight_dir)
    G_saved_folder = weight_dir+'G/'
    D_saved_folder = weight_dir+'D/'
    os.mkdir(G_saved_folder)
    os.mkdir(D_saved_folder)
    print('weight DIR CREATED')
except:
    print("weight DIR ALREADY EXIST")

try:
    os.mkdir(saved_img_dir)
    print('saved image DIR CREATED')
except:
    print("saved img DIR ALREADY EXIST")

try:
    copyfile('./config.py', result_dir+'config.py')
except:
    pass

# testing code
if testing_code:
    epochs = 40


# build crop dataset
crop_dataset       = build_CropDataset(image_size=crop_size, batch_size=gan_batch_size, img_folder=crop_folder)
crop_train_dataset = crop_dataset.train(augmentation, transform_rate=0.75)


# build dataset
dataset       = build_dataset(image_size=image_size, batch_size=batch_size)
train_dataset = dataset.train(train_dir, augmentation)
valid_dataset = dataset.test(valid_dir)
test_dataset  = dataset.test(test_dir)


# Load model
# g_model stands for the pre-trained Efficient model used in G
G = PAGenerator(model=g_model, input_shape=image_size) 
D = Discriminator(model=model, input_shape=crop_size)


# try load weight
try:
    print('try loading weights...')
    G.load_state_dict(torch.load(G_path), strict=True)
    D.load_state_dict(torch.load(D_path), strict=True)
    print('Load weight success !!!')
except:
    print('Train from scratch !!!')

G = G.to(device)
D = D.to(device)


# training process
gan_train = train_by_epoch(testing_code, 
                           G, 
                           D, 
                           epochs, 
                           gan_batch_size, 
                           train_dataset, 
                           valid_dataset,
                           crop_train_dataset,
                           device,
                           result_dir,
                           weight_dir,
                           classifier_loss=BCEloss,
                           gan_learning_rate=gan_lr,
                           classifier_learning_rate=learning_rate,
                           weight_clipping=weight_clipping, 
                           GAN_loss=GAN_loss, 
                           gradient_panelty=gradient_panelty, 
                           original_gan=original_gan
                           )
gan_train.train()

