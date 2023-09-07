import cv2
import glob
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np

#################### full size image dataset ####################


class LoadData(Dataset):
    def __init__(self, 
                 img_folder,
                 preprocessing=None,
                 transform=None,
                 transform_rate=0.75,
                 ):
        
        
        self.transform      = transform
        self.img_folder     = img_folder
        self.preprocessing  = preprocessing
        self.transform_rate = transform_rate

        # should be changed to your own folder name
        self.fra_img_paths  = sorted(glob.glob(self.img_folder+'fra/*.png'))
        self.nor_img_paths  = sorted(glob.glob(self.img_folder+'no/*.png'))
        self.fra_number     = len(self.fra_img_paths)
        self.nor_number     = len(self.nor_img_paths)

        # build image path and label list, fra = 1, nor = 0
        self.img_paths = []
        self.labels    = []

        for fra_img_path in self.fra_img_paths:
            self.img_paths.append(fra_img_path)
            self.labels.append(1)

        for nor_img_path in self.nor_img_paths:
            self.img_paths.append(nor_img_path)
            self.labels.append(0)
        
        
    def __len__(self):
        return len(self.img_paths)

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path  = self.img_paths[idx]
        label     = self.labels[idx]
        img       = cv2.imread(img_path)
        
        if self.preprocessing:
            img   = self.preprocessing(img)
        
        if self.transform:
            if np.random.uniform() < self.transform_rate:
                img = self.transform(img)

        
                
        # img = img.to(torch.float)
        img = transforms.ToTensor()(img)
        
        
        # return image and label dict
        return {'image': img, 'label': label}
    
    
    
class build_dataset():
    def __init__(self, image_size, batch_size):
        self.image_size    = image_size
        self.batch_size    = batch_size
        self.preprocessing = transforms.Compose([Rescale(self.image_size)])
        
    def train(self, img_folder, augmentation, transform_rate=0.75):

        train_dataset = LoadData(
            img_folder=img_folder,
            preprocessing=self.preprocessing, 
            transform=augmentation, 
            transform_rate=transform_rate
            )
        
        data_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=True, 
            num_workers=0, 
            drop_last=True
            )
        fra_number = train_dataset.fra_number
        nor_number = train_dataset.nor_number
        print(f'training data\nfra:{fra_number}  nor:{nor_number}')
        return data_loader
    
    
    def valid(self, img_folder):
        test_dataset = LoadData(img_folder=img_folder, preprocessing=self.preprocessing)
        data_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size,
            shuffle=False, 
            num_workers=0
            )
        fra_number = test_dataset.fra_number
        nor_number = test_dataset.nor_number
        print(f'valid data\nfra:{fra_number}  nor:{nor_number}')
        return data_loader
        
    
    def test(self, img_folder):
        test_dataset = LoadData(img_folder=img_folder, preprocessing=self.preprocessing)
        data_loader = DataLoader(
            test_dataset, 
            batch_size=1,
            shuffle=False, 
            num_workers=0
            )
        fra_number = test_dataset.fra_number
        nor_number = test_dataset.nor_number
        print(f'test data\nfra:{fra_number}  nor:{nor_number}')
        return data_loader
    
    

#################### patch dataset ####################


class LoadCropImage(Dataset):
    def __init__(self, 
                 img_folder,
                 preprocessing=None,
                 transform=None,
                 transform_rate=0.85,
                 ):
        
        
        self.transform = transform
        self.img_folder = img_folder
        self.img_paths = glob.glob(self.img_folder+'*.png')
        self.preprocessing  = preprocessing
        self.transform_rate = transform_rate
        
        
    def __len__(self):
        return len(self.img_paths)

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path  = self.img_paths[idx]
        img       = cv2.imread(img_path)
        
        if self.preprocessing:
            img   = self.preprocessing(img)
        
        if self.transform:
            if np.random.uniform() < self.transform_rate:
                img = self.transform(img)
                
        img = transforms.ToTensor()(img)*2-1

        return img
    
    
    
class build_CropDataset():
    def __init__(self, image_size, batch_size, img_folder):
        self.image_size    = image_size
        self.batch_size    = batch_size
        self.img_folder    = img_folder
        self.preprocessing = transforms.Compose([Rescale(self.image_size)])
        
    def train(self, augmentation, transform_rate=0.75):
        train_dataset = LoadCropImage(self.img_folder, 
                                      preprocessing=self.preprocessing, 
                                      transform=augmentation, 
                                      transform_rate=transform_rate
                                      )
        
        data_loader = DataLoader(train_dataset, 
                                 batch_size=self.batch_size,
                                 shuffle=True, 
                                 num_workers=0)
        
        print(f'training data number: {len(train_dataset)}')
        return data_loader
    
    
    def test(self):
        test_dataset = LoadCropImage(self.img_folder, preprocessing=self.preprocessing)
        data_loader = DataLoader(test_dataset, 
                                 batch_size=1,
                                 shuffle=False, 
                                 num_workers=0)
        return data_loader




#################### augmentation ####################

class Rescale():
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, img):
        
        if len(img.shape) < 3:
            img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        
        h, w          = img.shape[:2]
        channel       = img.shape[-1]
        scale         = min(self.output_size/w, self.output_size/h)
        new_h, new_w  = int(scale*h), int(scale*w)
        image_resized = cv2.resize(img, (new_w, new_h))
        pad_h, pad_w  = (self.output_size-new_h) // 2, (self.output_size-new_w) // 2
        
        image_paded   = np.full(shape=[self.output_size, self.output_size, channel], fill_value=0)
        image_paded[pad_h:new_h+pad_h, pad_w:new_w+pad_w, :] = image_resized
        image_paded   = np.array(image_paded, np.uint8)
        
        return image_paded
    
    
    
    
# augmentation CV2
class RandomClahe():
    def __init__(self, clipLimit=3, tileGridSize=(3,3), random=0.5):
        if np.random.uniform() < random:
            clipLimit+=3*(np.random.uniform()-0.5)
            
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    def __call__(self, img):
        if len(img.shape) < 3:
            img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        
        img[...,0] = self.clahe.apply(img[...,0])
        img[...,1] = self.clahe.apply(img[...,1])
        img[...,2] = self.clahe.apply(img[...,2])
        
        return img
    
    

    
# augmentation torch transforms
class RandomBright():
    def __init__(self, 
                 bright=1,
                 random_range=0.1,
                 aug_prob=0.25):
        
        bright_delta  = 2*(np.random.uniform(0, random_range)-random_range/2)
        self.bright   = bright+bright_delta
        self.aug_prob = aug_prob
        
    def __call__(self, img):
        if np.random.uniform() < self.aug_prob:
            img = transforms.functional.adjust_brightness(img, self.bright)
            
        return img
    
    
    
    
class RandomContrast():
    def __init__(self, 
                 contrast=1,
                 random_range=0.1,
                 aug_prob=0.25):
        
        delta = 2*(np.random.uniform(0, random_range)-random_range/2)
        self.contrast   = contrast+delta
        self.aug_prob = aug_prob
        
    def __call__(self, img):
        if np.random.uniform() < self.aug_prob:
            img = transforms.functional.adjust_contrast(img, self.contrast)
            
        return img
    
    
    
    
class RandomGamma():
    def __init__(self, 
                 gamma=1,
                 random_range=0.1,
                 aug_prob=0.25):
        
        delta = 2*(np.random.uniform(0, random_range)-random_range/2)
        self.gamma    = gamma+delta
        self.aug_prob = aug_prob
        
    def __call__(self, img):
        if np.random.uniform() < self.aug_prob:
            img = transforms.functional.adjust_gamma(img, self.gamma)
            
        return img
    
    
    
    
class RandomHue():
    def __init__(self, 
                 hue=0,
                 random_range=0.1,
                 aug_prob=0.25):
        
        delta = 2*(np.random.uniform(0, random_range)-random_range/2)
        self.hue      = hue+delta
        self.aug_prob = aug_prob
        
    def __call__(self, img):
        if np.random.uniform() < self.aug_prob:
            img = transforms.functional.adjust_hue(img, self.hue)
            
        return img
    
    
    
    
class RandomSaturation():
    def __init__(self, 
                 saturation=1,
                 random_range=0.1,
                 aug_prob=0.25):
        
        delta = 2*(np.random.uniform(0, random_range)-random_range/2)
        self.saturation = saturation+delta
        self.aug_prob   = aug_prob
        
    def __call__(self, img):
        if np.random.uniform() < self.aug_prob:
            img = transforms.functional.adjust_saturation(img, self.saturation)
            
        return img