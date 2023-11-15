from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import InterpolationMode
from pytorch_lightning.demos.mnist_datamodule import MNIST
from torch.utils.data import DataLoader, random_split

from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl
#from utils import *
from PIL import Image
import radiogalaxies_bnns.datasets.mirabest as mirabest

#config_dict, config = parse_config('config1.txt')

#data
#transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((datamean ,), (datastd,))])

class MNISTDataModule(pl.LightningDataModule):
    
    def __init__(self, batch_size, hmc, DATASETS_PATH = Path('./MNISTdataset')):
        super().__init__()
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset = MNIST(DATASETS_PATH, train = True, download = True, transform = transform)
        self.mnist_test = MNIST(DATASETS_PATH, train = False, download = True, transform = transform)
        self.mnist_train, self.mnist_val = random_split(dataset, [50000, 10000])
        self.batch_size = batch_size
        
        if(hmc == True):
            self.batch_size_train = 50000
            self.batch_size_val = 10000
            self.batch_size_test = 10000
        else: 
            self.batch_size_train = self.batch_size
            self.batch_size_val = self.batch_size
            self.batch_size_test = self.batch_size
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size = self.batch_size_train)    
        #return DataLoader(self.mnist_train, num_workers = 4,prefetch_factor = 2, pin_memory = True, batch_size = self.batch_size_train)    

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size = self.batch_size_val)    
        #return DataLoader(self.mnist_val, num_workers = 4, prefetch_factor = 2, pin_memory = True, batch_size = self.batch_size_val)    

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size = self.batch_size_test)
        #return DataLoader(self.mnist_val, num_workers = 4, prefetch_factor = 2, pin_memory = True, batch_size = self.batch_size_test)    

    #why is this function required
    def predict_dataloader(self):
        return DataLoader(self.mnist_test, batch_size = self.batch_size_test)
        #return DataLoader(self.mnist_test, num_workers = 4, prefetch_factor = 2, pin_memory = True, batch_size = self.batch_size_test)


class MiraBestDataModule(pl.LightningDataModule):
    
    def __init__(self, config_dict, hmc): #, config):
        super().__init__()

        self.batch_size = config_dict['training']['batch_size']
        self.validation_split = config_dict['training']['frac_val']
        self.dataset = config_dict['data']['dataset']
        self.path = Path(config_dict['data']['datadir'])
        self.datamean = config_dict['data']['datamean']
        self.datastd = config_dict['data']['datastd']
        self.augment = 'False' #config_dict['data']['augment'] #more useful to define while calling train/val loader?
        self.imsize = config_dict['training']['imsize']

        if(hmc == True):
            self.batch_size_train = 584
            self.batch_size_val = 145
            self.batch_size_test = 104
        else: 
            self.batch_size_train = self.batch_size
            self.batch_size_val = self.batch_size
            self.batch_size_test = self.batch_size

    
        
    def transforms(self, aug):
        
        if(aug == 'False'):
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((self.datamean ,), (self.datastd,))])

        else:
            print("AUGMENTING")                           
            #crop, pad(reflect), rotate, to tensor, normalise
            #change transform to transform_aug for the training and validation set only:train_data_confident
            crop     = transforms.CenterCrop(self.imsize)
            #pad      = transforms.Pad((0, 0, 1, 1), fill=0)
            
            
            transform = transforms.Compose([crop,
                                            #pad,
                                            transforms.RandomRotation(360, interpolation=InterpolationMode.BILINEAR, expand=False),
                                            transforms.ToTensor(),
                                            transforms.Normalize((self.datamean ,), (self.datastd,)),
                                            ])
            
        return transform
    
    def train_val_loader(self, set):
        
        transform = self.transforms(self.augment)

        if(self.dataset =='MBFRConf'):
            train_data_confident = mirabest.MBFRConfident(self.path, train=True,
                                                          transform=transform, target_transform=None,
                                                          download=True)
            train_data_conf = train_data_confident
        
        
        elif(self.dataset == 'MBFRConf+Uncert'):
            train_data_confident = mirabest.MBFRConfident(self.path, train=True,
                             transform=transform, target_transform=None,
                             download=True)
            
            train_data_uncertain = mirabest.MBFRUncertain(self.path, train=True,
                             transform=transform, target_transform=None,
                             download=True)
            
            #concatenate datasets
            train_data_conf= torch.utils.data.ConcatDataset([train_data_confident, train_data_uncertain])
            
                
        #train-valid
        dataset_size = len(train_data_conf)
        indices = list(range(dataset_size))
        split = int(dataset_size*0.2) #int(np.floor(validation_split * dataset_size))
        shuffle_dataset = True
        random_seed = 15

        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
    
        # Creating data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
         
        train_loader = torch.utils.data.DataLoader(train_data_conf, batch_size=self.batch_size_train, sampler=train_sampler)
        validation_loader = torch.utils.data.DataLoader(train_data_conf, batch_size=self.batch_size_val, sampler=valid_sampler)
        
        if(set == 'train'):
            return train_loader
        elif(set == 'val'):
            return validation_loader #, train_sampler, valid_sampler
    
    def train_dataloader(self):
        return  self.train_val_loader('train')

    def val_dataloader(self):
        return self.train_val_loader('val')

    #add hybrids
    def test_dataloader(self):           
        #no augmentation for test_loader
        
        transform = self.transforms(aug='False')
        
        if(self.dataset =='MBFRConf'):
 
            test_data_confident = mirabest.MBFRConfident(self.path, train=False,
                                                         transform=transform, target_transform=None,
                                                         download=False)
            test_data_conf = test_data_confident

            
        elif(self.dataset == 'MBFRConf+Uncert'):

            #confident
            test_data_confident = mirabest.MBFRConfident(self.path, train=False,
                             transform=transform, target_transform=None,
                             download=True)
            
            #uncertain
            test_data_uncertain = mirabest.MBFRUncertain(self.path, train=False,
                             transform=transform, target_transform=None,
                             download=True)
            
            #concatenate datasets
            test_data_conf = torch.utils.data.ConcatDataset([test_data_confident, test_data_uncertain])
        test_loader = torch.utils.data.DataLoader(dataset=test_data_conf, batch_size=self.batch_size_test,shuffle=False)
        
        return test_loader
    
def testloader_mb_uncert(test_data_uncert, path):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.0031 ,), (0.0350,))])

    #options for test_data and test_data1
    if(test_data_uncert == 'MBFRConfident'):
        # confident
        test_data = mirabest.MBFRConfident(path, train=False,
                         transform=transform, target_transform=None,
                         download=False)
        test_dataloader =  torch.utils.data.DataLoader(dataset=test_data, batch_size = 104, shuffle=False)

        
        test_data1 = mirabest.MBFRConfident(path, train=False,
                         transform=None, target_transform=None,
                         download=False)
        data_type = 'Conf'
    elif(test_data_uncert == 'MBFRUncertain'):
        # uncertain
        
        test_data = mirabest.MBFRUncertain(path, train=False,
                         transform=transform, target_transform=None,
                         download=False)
        test_dataloader =  torch.utils.data.DataLoader(dataset=test_data, batch_size = 49, shuffle=False)
        
        test_data1 = mirabest.MBFRUncertain(path, train=False,
                         transform=None, target_transform=None,
                         download=False)
        data_type = 'Uncert'
    elif(test_data_uncert == 'MBHybrid'):
        #hybrid
        test_data = mirabest.MBHybrid(path, train=True,
                         transform=transform, target_transform=None,
                         download=False)
        test_dataloader =  torch.utils.data.DataLoader(dataset=test_data, batch_size = 30, shuffle=False)

        test_data1 = mirabest.MBHybrid(path, train=True,
                         transform=None, target_transform=None,
                         download=False)
        data_type = 'Hybrid'
    else:
        print("Test data for uncertainty quantification misspecified")
    # test_dataloader =  torch.utils.data.DataLoader(dataset=test_data, batch_size = , shuffle=False)
    return test_dataloader, test_data1, data_type, test_data
