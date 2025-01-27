import matplotlib.pyplot as plt
import random
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import cv2
import itertools

# Định sẵn đường đẫn đến các dataset
DEFAULT_LIVESTOCK_DIR = "./data/livestock/part_III_cropped"
DEFAULT_MVTEC_DIR = "E:/UnitWTF/lab ai/mvtec_anomaly_detection/wood"
DEFAULT_MIAD_DIR = "E:/UnitWTF/dataset/photovoltaic_module"
DEFAULT_UBC_DIR = "E:/UBC-OCEAN/train_images"
# Traning Dataset for livestock
class LivestockTrainDataset(Dataset):
    def __init__(self, img_size, fake_dataset_size):

        if os.path.isdir(DEFAULT_LIVESTOCK_DIR):
            self.img_dir = os.path.join(DEFAULT_LIVESTOCK_DIR, "Train") # set image dir
        else:
            self.img_dir = UNDEFINE # set undefine if not found

        #get a list of image path
        self.img_files = list(
                            np.random.choice(
                                [os.path.join(self.img_dir, img)
                            for img in os.listdir(self.img_dir)
                            if (os.path.isfile(os.path.join(self.img_dir,
                            img)) and img.endswith('jpg'))],
                            size=fake_dataset_size)
                            )
        
        # Tuỳ chỉnh độ dài data
        self.fake_dataset_size = fake_dataset_size # needed otherwise there are
        # 125000 images, and this is too much

        #Augmentation setting
        self.transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.PILToTensor(),
            transforms.Lambda(lambda img: img.float()),
            transforms.Lambda(lambda img: img / 255.)
        ]) 
        #number of image
        self.nb_img = len(self.img_files)
        #number of (color) channel
        self.nb_channels = 3

    #get length of data
    def __len__(self):
        return max(self.nb_img, self.fake_dataset_size)

    #get specific item in the dataset via index
    def __getitem__(self, index):
        index = index % self.nb_img
        img = Image.open(self.img_files[index])
        
        return self.transform(img), 1 # one if the ground truth if there is one

class LivestockTestDataset(Dataset):
    def __init__(self, img_size, fake_dataset_size):
        if os.path.isdir(DEFAULT_LIVESTOCK_DIR):
            self.img_dir = os.path.join(DEFAULT_LIVESTOCK_DIR, "Test")
        else:
            self.img_dir = UNDEFINE
        self.img_files = list(
                            np.random.choice(
                                [os.path.join(self.img_dir, img)
                            for img in os.listdir(self.img_dir)
                            if (os.path.isfile(os.path.join(self.img_dir, img))
                            and img.endswith('.jpg'))],
                            size=fake_dataset_size)
                            )
        self.fake_dataset_size = fake_dataset_size # needed otherwise there are
        self.gt_files = [s.replace(".jpg", "_gt.png") for s in self.img_files]
        self.transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.PILToTensor(),
            transforms.Lambda(lambda img: img.float()),
            transforms.Lambda(lambda img: img / 255.)
        ]) 
        self.nb_img = len(self.img_files) # recompute the size,
        # fake_dataset_size may have changed it
        self.nb_channels = 3

    def __len__(self):
        return self.fake_dataset_size

    def __getitem__(self, index):
        img = Image.open(self.img_files[index])
        gt = Image.open(self.gt_files[index])

        return self.transform(img), self.transform(gt)


class MVTecTrainDataset(Dataset):
    def __init__(self, img_size, fake_dataset_size):
        if os.path.isdir(DEFAULT_MVTEC_DIR):
            self.img_dir = os.path.join(DEFAULT_MVTEC_DIR, "train", "good")
        else:
            self.img_dir = UNDEFINE
        self.img_files = list(
                            np.random.choice(
                                [os.path.join(self.img_dir, img)
                            for img in os.listdir(self.img_dir)
                            if (os.path.isfile(os.path.join(self.img_dir,
                            img)) and img.endswith('png'))],
                            size=fake_dataset_size)
                            )
        self.fake_dataset_size = fake_dataset_size # needed otherwise there are
        # 125000 images, and this is too much
        self.transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.PILToTensor(),
            transforms.Lambda(lambda img: img.float()),
            transforms.Lambda(lambda img: img / 255.)
        ]) 
        self.nb_img = len(self.img_files)
        self.nb_channels = 3

    def __len__(self):
        return max(self.nb_img, self.fake_dataset_size)

    def __getitem__(self, index):
        index = index % self.nb_img
        img = Image.open(self.img_files[index])
        
        return self.transform(img), 1 # one if the ground truth if there is one

class MVTecTestDataset(Dataset):
    def __init__(self, img_size, fake_dataset_size):
        if os.path.isdir(DEFAULT_MVTEC_DIR):
            self.img_dir = os.path.join(DEFAULT_MVTEC_DIR, "test", "hole")
        else:
            self.img_dir = UNDEFINE
        self.img_files = list(
                            np.random.choice(
                                [os.path.join(self.img_dir, img)
                            for img in os.listdir(self.img_dir)
                            if (os.path.isfile(os.path.join(self.img_dir, img))
                            and img.endswith('.png'))],
                            size=fake_dataset_size)
                            )
        self.fake_dataset_size = fake_dataset_size # needed otherwise there are
        self.gt_files = [s.replace(".png", "_mask.png").replace("test","ground_truth") for s in self.img_files]
        self.transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.PILToTensor(),
            transforms.Lambda(lambda img: img.float()),
            transforms.Lambda(lambda img: img / 255.)
        ]) 
        self.nb_img = len(self.img_files) # recompute the size,
        # fake_dataset_size may have changed it
        self.nb_channels = 3

    def __len__(self):
        return self.fake_dataset_size

    def __getitem__(self, index):
        img = Image.open(self.img_files[index])
        gt = Image.open(self.gt_files[index])

        return self.transform(img), self.transform(gt)


class MVTecTrainDataset(Dataset):
    def __init__(self, img_size, fake_dataset_size):
        if os.path.isdir(DEFAULT_MVTEC_DIR):
            self.img_dir = os.path.join(DEFAULT_MVTEC_DIR, "train", "good")
        else:
            self.img_dir = UNDEFINE
        self.img_files = list(
                            np.random.choice(
                                [os.path.join(self.img_dir, img)
                            for img in os.listdir(self.img_dir)
                            if (os.path.isfile(os.path.join(self.img_dir,
                            img)) and img.endswith('png'))],
                            size=fake_dataset_size)
                            )
        self.img_size = img_size
        self.fake_dataset_size = fake_dataset_size # needed otherwise there are
        # 125000 images, and this is too much
        self.transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.PILToTensor(),
            transforms.Lambda(lambda img: img.float()),
            transforms.Lambda(lambda img: img / 255.)
        ]) 
        self.nb_img = len(self.img_files)
        self.nb_channels = 3

    def __len__(self):
        return max(self.nb_img, self.fake_dataset_size)

    def __getitem__(self, index):
        index = index % self.nb_img
        # img = cv2.imread(self.img_files[index])
        
        # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        # img = self.transform(img)
        img = Image.open(self.img_files[index])
        
        return self.transform(img), 1 # one if the ground truth if there is one

class MIADTestDataset(Dataset):
    def __init__(self, img_size, fake_dataset_size):
        if os.path.isdir(DEFAULT_MIAD_DIR):
            self.img_dir = os.path.join(DEFAULT_MIAD_DIR, "test", "broken")
        else:
            self.img_dir = UNDEFINE
        self.img_files = list(
                            np.random.choice(
                                [os.path.join(self.img_dir, img)
                            for img in os.listdir(self.img_dir)
                            if (os.path.isfile(os.path.join(self.img_dir, img))
                            and img.endswith('.png'))],
                            size=fake_dataset_size)
                            )
        self.fake_dataset_size = fake_dataset_size # needed otherwise there are
        self.gt_files = [s.replace(".png", "_mask.png").replace("test","ground_truth") for s in self.img_files]
        self.transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.PILToTensor(),
            transforms.Lambda(lambda img: img.float()),
            transforms.Lambda(lambda img: img / 255.)
        ]) 
        self.nb_img = len(self.img_files) # recompute the size,
        # fake_dataset_size may have changed it
        self.nb_channels = 3

    def __len__(self):
        return self.fake_dataset_size

    def __getitem__(self, index):
        img = Image.open(self.img_files[index])
        gt = Image.open(self.gt_files[index])

        return self.transform(img), self.transform(gt)

class MIADTrainDataset(Dataset):
    def __init__(self, img_size, fake_dataset_size, all_in = False):
        if os.path.isdir(DEFAULT_MIAD_DIR):
            self.img_dir = os.path.join(DEFAULT_MIAD_DIR, "train", "good")
        else:
            self.img_dir = UNDEFINE
        print("all_in")
        # if not all_in:

        self.img_files = list(
                            np.random.choice(
                                [os.path.join(self.img_dir, img)
                            for img in os.listdir(self.img_dir)
                            if (os.path.isfile(os.path.join(self.img_dir,
                            img)) and img.endswith('png'))],
                            size=fake_dataset_size)
        )
        # else:                        
        # self.img_files = list(
                            
        #                         [os.path.join(self.img_dir, img)
        #                     for img in os.listdir(self.img_dir)
        #                     if (os.path.isfile(os.path.join(self.img_dir,
        #                     img)) and img.endswith('png'))]
        #                     )
        # self.fake_dataset_size = fake_dataset_size # needed otherwise there are
        # 125000 images, and this is too much
        self.transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.PILToTensor(),
            transforms.Lambda(lambda img: img.float()),
            transforms.Lambda(lambda img: img / 255.)
        ]) 
        self.nb_img = len(self.img_files)
        print("the number of image is:", self.nb_img)
        self.nb_channels = 3

    def __len__(self):
        return self.nb_img
    def __getitem__(self, index):
        index = index % self.nb_img
        
        img = Image.open(self.img_files[index])
        
        return self.transform(img), 1 # one if the ground truth if there is one
  
class UBCTestDataset(Dataset):
    def __init__(self, img_size, fake_dataset_size):
        if os.path.isdir(DEFAULT_UBC_DIR):
            self.img_dir = os.path.join(DEFAULT_UBC_DIR, "EC")
            
            self.img_fol_dir =list(
                            np.random.choice( list(os.path.join(self.img_dir, img_fol) for img_fol in os.listdir(self.img_dir)),
                            size=47)
                            )
        else:
            self.img_dir = UNDEFINE
        self.img_files = list(
                            np.random.choice(list(itertools.chain.from_iterable([[os.path.join(img_fol, img)
                            for img in os.listdir(img_fol)  
                            if (os.path.isfile(os.path.join(img_fol, img))
                            and img.endswith('.png') and "_" in img
                            and os.path.isfile(os.path.join(os.path.abspath(os.path.join(img_fol, "../../..")), 
                                              "global_images", 
                                              img.split("_")[0]+"_global.png")))] for img_fol in self.img_fol_dir])),
                            size=fake_dataset_size)
        )
        self.fake_dataset_size = fake_dataset_size # needed otherwise there are
        self.global_img_files = [os.path.join(os.path.abspath(os.path.join(s, "../../../..")), "global_images", os.path.basename(s).split("_")[0]+"_global.png") for s in self.img_files]

        # self.gt_files = [s.replace(".png", "_mask.png").replace("test","ground_truth") for s in self.img_files]
        self.transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.PILToTensor(),
            transforms.Lambda(lambda img: img.float()),
            transforms.Lambda(lambda img: img / 255.)
        ]) 
        self.nb_img = len(self.img_files) # recompute the size,
        # fake_dataset_size may have changed it
        self.nb_channels = 3

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_loc= Image.open(self.img_files[index])
        img_glo = Image.open(self.global_img_files[index])
        # gt = Image.open(self.gt_files[index])

        return self.transform(img_loc), self.transform(img_glo), 0
    
class UBCTestPlotDataset(Dataset):
    def __init__(self, category, img_size, fake_dataset_size):
        if os.path.isdir(DEFAULT_UBC_DIR):
            self.img_dir = os.path.join(DEFAULT_UBC_DIR, category)
            
            self.img_fol_dir =list(
                            np.random.choice( list(os.path.join(self.img_dir, img_fol) for img_fol in os.listdir(self.img_dir)),
                            size=47)
                            )
        else:
            self.img_dir = UNDEFINE
        self.img_files = list(itertools.chain.from_iterable([[os.path.join(img_fol, img)
                            for img in os.listdir(img_fol)  
                            if (os.path.isfile(os.path.join(img_fol, img))
                            and img.endswith('.png') and "_" in img
                            and os.path.isfile(os.path.join(os.path.abspath(os.path.join(img_fol, "../../..")), 
                                              "global_images", 
                                              img.split("_")[0]+"_global.png")))] for img_fol in self.img_fol_dir]))
        self.fake_dataset_size = fake_dataset_size # needed otherwise there are
        self.global_img_files = [os.path.join(os.path.abspath(os.path.join(s, "../../../..")), "global_images", os.path.basename(s).split("_")[0]+"_global.png") for s in self.img_files]

        # self.gt_files = [s.replace(".png", "_mask.png").replace("test","ground_truth") for s in self.img_files]
        self.transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.PILToTensor(),
            transforms.Lambda(lambda img: img.float()),
            transforms.Lambda(lambda img: img / 255.)
        ]) 
        self.nb_img = len(self.img_files) # recompute the size,
        # fake_dataset_size may have changed it
        self.nb_channels = 3

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_loc= Image.open(self.img_files[index])
        img_glo = Image.open(self.global_img_files[index])
     
        # gt = Image.open(self.gt_files[index])

        return self.transform(img_loc), self.transform(img_glo), 0, self.img_files[index], os.path.basename(self.img_files[index]).split("_")[0] , os.path.basename(self.img_files[index])

class UBCTrainDataset(Dataset):
    def __init__(self, img_size, fake_dataset_size, all_in = False):
        if os.path.isdir(DEFAULT_UBC_DIR):
            self.img_dir = os.path.join(DEFAULT_UBC_DIR, "MC")
            self.img_fol_dir = list(
                            np.random.choice(list(os.path.join(self.img_dir, img_fol) for img_fol in os.listdir(self.img_dir)),
                            size=45)
                            )
        else:
            self.img_dir = UNDEFINE
      
        # if not all_in:

        self.img_files = list(itertools.chain.from_iterable([[os.path.join(img_fol, img)
                            for img in os.listdir(img_fol)  
                            if (os.path.isfile(os.path.join(img_fol, img))
                            and img.endswith('.png') and "_" in img 
                            and os.path.isfile(os.path.join(os.path.abspath(os.path.join(img_fol, "../../..")), 
                                              "global_images", 
                                              img.split("_")[0]+"_global.png")))] for img_fol in self.img_fol_dir]))
        
        self.global_img_files = [os.path.join(os.path.abspath(os.path.join(s, "../../../..")), 
                                              "global_images", 
                                              os.path.basename(s).split("_")[0]+"_global.png") for s in self.img_files]

        # else:                        
        # self.img_files = list(
                            
        #                         [os.path.join(self.img_dir, img)
        #                     for img in os.listdir(self.img_dir)
        #                     if (os.path.isfile(os.path.join(self.img_dir,
        #                     img)) and img.endswith('png'))]
        #                     )
        # self.fake_dataset_size = fake_dataset_size # needed otherwise there are
        # 125000 images, and this is too much
        self.transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.PILToTensor(),
            transforms.Lambda(lambda img: img.float()),
            transforms.Lambda(lambda img: img / 255.)
        ]) 
        self.nb_img = len(self.img_files)
        print("the number of image is:", self.nb_img)
        self.nb_channels = 3

    def __len__(self):
        return self.nb_img
    def __getitem__(self, index):
        index = index % self.nb_img
        
        img_loc = Image.open(self.img_files[index])
        img_glo = Image.open(self.global_img_files[index])
        
        return self.transform(img_loc), self.transform(img_glo), 1 # one if the ground truth if there is one
    

