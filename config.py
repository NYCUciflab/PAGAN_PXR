from utils.dataset import *


##### INFORMATION #####
testing_code     = False
annotaton        = 'test_1'
device           = 'cuda:0'
model_name       = 'efficientnet_b0'
epochs           = 100
learning_rate    = 0.0001
batch_size       = 8
image_size       = 512

# image folder
train_dir   = 'ENTER YOUR TRAINING FORDER PATH'
valid_dir   = 'ENTER YOUR VALIDATION FORDER PATH'
test_dir    = 'ENTER YOUR TEST FORDER PATH'
crop_folder = 'ENTER YOUR CROP FOLDER PATH'


# GAN
gan_lr           = 0.00005
weight_clipping  = False
crop_size        = 128
gan_batch_size   = 8
GAN_loss         = 'lsgan'
gan_pre_trained  = True
gradient_panelty = 30
original_gan     = True

# pre-trained weights
D_path           = None
G_path           = None
pre_trained      = 'imagenet'
pre_trained_path = None




##### AUGMENTATION #####

# CLAHE
clipLimit      = 5
tileGridSize   = (3,3)
clahe_aug_prob = 0.5
# BRIGHT
bright_range    = 0.12
bright_aug_prob = 0.75
# CONTRAST
contrast_range    = 0.12
contrast_aug_prob = 0.75
# GAMMA
gamma_range    = 0.15
gamma_aug_prob = 0.5
# HUE
hue_range    = 0.12
hue_aug_prob = 0.5
# SATURATION
sat_range    = 0.12
sat_aug_prob = 0.5
# ROTATION
rotate_angle = 25