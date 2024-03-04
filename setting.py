from config import *

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