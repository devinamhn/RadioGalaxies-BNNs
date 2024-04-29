import argparse
import configparser as ConfigParser 
import ast
import torch
import numpy as np 
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler
import torch.nn.functional as F
import torchvision
from pathlib import Path

from radiogalaxies_bnns.datasets import mirabest
from radiogalaxies_bnns.eval.uncertainty.uncertainty import entropy_MI, calibration
from galaxy_mnist import GalaxyMNIST, GalaxyMNISTHighrez
from radiogalaxies_bnns.inference.datamodules import MNISTDataModule, MiraBestDataModule, testloader_mb_uncert

from cata2data import CataData
from torch.utils.data import DataLoader
from radiogalaxies_bnns.datasets.mightee import MighteeZoo
import radiogalaxies_bnns.inference.utils as utils

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def get_dataloader(test_data_uncert, path):

    test_data = test_data_uncert
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.0031 ,), (0.0350,))])
    test_data1 = test_data_uncert
    
    if(test_data_uncert == 'MBFRConfident'):
        
        #confident test set
        #confident test set
        test_data = mirabest.MBFRConfident(path, train=False,
                            transform=transform, target_transform=None,
                            download=False)
        

    elif(test_data_uncert == 'Galaxy_MNIST'):
        transform = torchvision.transforms.Compose([ torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((150,150), antialias = True), 
        torchvision.transforms.Grayscale(),
        ])
        # 64 pixel images
        train_dataset = GalaxyMNISTHighrez(
            root='/share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/data/dataGalaxyMNISTHighres',
            download=True,
            train=True,  # by default, or set False for test set
            transform = transform
        )

        test_dataset = GalaxyMNISTHighrez(
            root='/share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/data/dataGalaxyMNISTHighres',
            download=True,
            train=False,  # by default, or set False for test set
            transform = transform
        )
        gal_mnist_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=104, shuffle = False)

        # for i, (x_test_galmnist, y_test_galmnist) in enumerate(gal_mnist_test_loader):
        #     x_test_galmnist, y_test_galmnist = x_test_galmnist.to(device), y_test_galmnist.to(device)
        #     y_test_galmnist = torch.zeros(104).to(device)
        #     if(i==0):
        #         break
        test_data = test_dataset

    elif(test_data_uncert == 'mightee'):
        transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(150),  # Rescale to adjust for resolution difference between MIGHTEE & RGZ - was 70
            torchvision.transforms.Normalize((1.59965605788234e-05,), (0.0038063037602458706,)),
        ]
        )
        paths = utils.Path_Handler()._dict()
        set = 'certain'

        data = MighteeZoo(path=paths["mightee"], transform=transform, set="certain")
        test_loader = DataLoader(data, batch_size=len(data))
        # for i, (x_test, y_test) in enumerate(test_loader):
        #     x_test, y_test = x_test.to(device), y_test.to(device)
                
        test_data = data

    return test_data

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    return img

path='/share/nas2/dmohan/RadioGalaxies-BNNs/dataMiraBest'
test_data_uncert ='Galaxy_MNIST'

# index = 0
test_data = get_dataloader(test_data_uncert, path)

for b, (im_batch, y_batch) in enumerate(DataLoader(test_data, batch_size=16)):
        
        fig = plt.figure(figsize=(13.0, 13.0))
        fig.subplots_adjust(0, 0, 1, 1)
        grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0)


        
        for i, (ax, im, y) in enumerate(zip(grid, list(im_batch), list(y_batch))):

            # im = im_batch[i].squeeze().numpy() 
            im = im.squeeze().numpy()
            
            ax.axis("off")
            # text = f"FR{y+1}"

            # ax.text(1, 66, text, fontsize=23, color="yellow")

            # # contours
            # threshold = 1
            # ax.contour(np.where(im > threshold, 1, 0), cmap="cool", alpha=0.1)

            ax.imshow(im)
        

        plt.axis("off")
        pil_img = fig2img(fig)
        plt.savefig(f"/share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/postage/gm1{b}.png")
        plt.close(fig)
        if(b == 0):
             break