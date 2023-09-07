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


# GAN
gan_lr           = 0.00005
weight_clipping  = False
crop_size        = 128
gan_batch_size   = 8
GAN_loss         = 'lsgan'
gan_pre_trained  = True
gradient_panelty = 30
D_path = None
G_path = None
original_gan = True


pre_trained      = 'imagenet'
pre_trained_path = None


train_dir = 'ENTER YOUR TRAINING FORDER PATH'
valid_dir = 'ENTER YOUR VALIDATION FORDER PATH'
test_dir  = 'ENTER YOUR TEST FORDER PATH'
crop_folder = 'ENTER YOUR CROP FOLDER PATH'


##### AUGMENTATION #####
"""
dataset = build_dataset(image_size=128, batch_size=16)
train_dataset = dataset.train(augmentation)
valid_dataset = dataset.test()
"""

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



augmentation = transforms.Compose([RandomClahe(clipLimit, tileGridSize, random=clahe_aug_prob),
                                   transforms.ToPILImage(),
                                   RandomBright(random_range=bright_range, aug_prob=bright_aug_prob),
                                   RandomContrast(random_range=contrast_range, aug_prob=contrast_aug_prob),
                                   RandomGamma(random_range=gamma_range, aug_prob=gamma_aug_prob),
                                   RandomHue(random_range=hue_range, aug_prob=hue_aug_prob),
                                   RandomSaturation(random_range=sat_range, aug_prob=sat_aug_prob),
                                   transforms.RandomRotation(rotate_angle)])



# define pre-trained weights
if pre_trained=='imagenet':
    weights = 'IMAGENET1K_V1'
else:
    weights = 'DEFAULT'
    print('train from scratch!!!')



# model
if model_name == 'efficientnet_b0':
    from torchvision.models import efficientnet_b0
    g_model = efficientnet_b0(weights='IMAGENET1K_V1')
    if pre_trained:
        model = efficientnet_b0(weights='IMAGENET1K_V1')
    else:
        model = efficientnet_b0(weights='DEFAULT')

elif model_name == 'efficientnet_b1':
    from torchvision.models import efficientnet_b1
    g_model = efficientnet_b1(weights='IMAGENET1K_V1')
    if pre_trained:
        model = efficientnet_b1(weights='IMAGENET1K_V1')
    else:
        model = efficientnet_b1(weights='DEFAULT')

elif model_name == 'efficientnet_b2':
    from torchvision.models import efficientnet_b2
    g_model = efficientnet_b2(weights='IMAGENET1K_V1')
    if pre_trained:
        model = efficientnet_b2(weights='IMAGENET1K_V1')
    else:
        model = efficientnet_b2(weights='DEFAULT')


elif model_name == 'efficientnet_b3':
    from torchvision.models import efficientnet_b3
    g_model = efficientnet_b3(weights='IMAGENET1K_V1')
    if pre_trained:
        model = efficientnet_b3(weights='IMAGENET1K_V1')
    else:
        model = efficientnet_b3(weights='DEFAULT')


elif model_name == 'efficientnet_b4':
    from torchvision.models import efficientnet_b4
    g_model = efficientnet_b4(weights='IMAGENET1K_V1')
    if pre_trained:
        model = efficientnet_b4(weights='IMAGENET1K_V1')
    else:
        model = efficientnet_b4(weights='DEFAULT')
