import torch
import sys

# sys.path.append('/share/nas2/dmohan/mcmc/hamildev/hamiltorch')
import hamiltorch

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torchvision.transforms as T

from cata2data import CataData
from torch.utils.data import DataLoader
from radiogalaxies_bnns.datasets.mightee import MighteeZoo
from radiogalaxies_bnns.inference.utils import Path_Handler

import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import InterpolationMode
from pytorch_lightning.demos.mnist_datamodule import MNIST
from torch.utils.data import DataLoader, random_split
from radiogalaxies_bnns.inference.datamodules import MNISTDataModule, MiraBestDataModule, testloader_mb_uncert
from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl
from radiogalaxies_bnns.inference import utils
from PIL import Image

from radiogalaxies_bnns.inference.models import MLP, LeNet

from matplotlib import pyplot as plt
from radiogalaxies_bnns.datasets import mirabest
from radiogalaxies_bnns.eval.uncertainty.uncertainty import entropy_MI, overlapping, GMM_logits, calibration
import csv
import seaborn as sns
# import arviz as az
# hamiltorch.set_random_seed(123)

def credible_interval(samples, credibility):
    '''
    calculate credible interval - equi-tailed interval instead of highest density interval

    samples values and indices
    '''
    mean_samples = samples.mean()
    sorted_samples = np.sort(samples)
    lower_bound = 0.5 * (1 - credibility)
    upper_bound = 0.5 * (1 + credibility)

    index_lower = int(np.round(len(samples) * lower_bound))
    index_upper = int(np.round(len(samples) * upper_bound))

    return sorted_samples, index_lower, index_upper, mean_samples

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = LeNet(1, 2)

path = '/share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/results/inits/'


params_hmc = torch.load(path+'/thin_chain'+str(0), map_location=torch.device(device)) #torch.load(path+'params_hmc_thin1000.pt', map_location = torch.device(device))

burn_in = 0
params_hmc = params_hmc[burn_in:]

tau_list = []
tau = 100. 


for w in model.parameters():
    tau_list.append(tau)
tau_list = torch.tensor(tau_list).to(device)

transform = T.Compose(
    [
        T.ToTensor(),
        T.Resize(150),  # Rescale to adjust for resolution difference between MIGHTEE & RGZ - was 70
        T.Normalize((1.59965605788234e-05,), (0.0038063037602458706,)),
    ]
)
paths = Path_Handler()._dict()
set = 'certain'

data = MighteeZoo(path=paths["mightee"], transform=transform, set="certain")
test_loader = DataLoader(data, batch_size=len(data))
for i, (x_test, y_test) in enumerate(test_loader):
    x_test, y_test = x_test.to(device), y_test.to(device)

print(len(data))
print(len(y_test))


# temp_list = {0: 1.0, 1:1e-1, 2:1e-2, 3:5e-3, 4:1e-4, 5:1e-5}
temp = 1

#get predictions
# pred_list, log_prob_list = hamiltorch.predict_model(model, x = x_test, y = y_test, samples=params_hmc, model_loss='multi_class_linear_output', tau_out=1., tau_list=tau_list, temp = temp)
pred_list, log_prob_list = hamiltorch.predict_model(model, x = x_test, y = y_test, samples=params_hmc, model_loss='multi_class_linear_output', tau_out=1., tau_list=tau_list)
_, pred = torch.max(pred_list, 2)

print(pred_list.shape) 

_, pred = torch.max(pred_list, 2)
acc = []
acc = torch.zeros( int(len(params_hmc))-1)
nll = torch.zeros( int(len(params_hmc))-1)
ensemble_proba = F.softmax(pred_list[0], dim=-1)

for s in range(1,len(params_hmc)):

    _, pred = torch.max(pred_list[:s].mean(0), -1)
    acc[s-1] = (pred.float() == y_test.flatten()).sum().float()/y_test.shape[0]
    ensemble_proba += F.softmax(pred_list[s], dim=-1) #ensemble prob is being added - save all the softmax values
    nll[s-1] = F.nll_loss(torch.log(ensemble_proba.cpu()/(s+1)), y_test[:].long().cpu().flatten(), reduction='mean')
    
print('ACCURACY', torch.mean(acc), '+/-',  torch.std(acc))
print('NLL', torch.mean(nll), '+/-', torch.std(nll))





avg_error_mean = []

error_all = []
entropy_all = []
mi_all = []
aleat_all = []

burn = 0
color = []
for k in range(len(y_test)):
    # print('galaxy', k)

    x = [0, 1]
    x = np.tile(x, (len(params_hmc)-burn, 1))

    target = y_test[k].cpu().detach().numpy()
    if(target == 0):
        label = 'FRI'
        color.append('C1')
        
    elif(target ==1):
        label = 'FRII'
        color.append('C2')
    
    # print(label)
    
    logits = pred_list[burn:][:, k:(k+1), :2].cpu().detach().numpy()
    softmax_values = F.softmax(pred_list[burn:][:, k:(k+1), :2], dim =-1).cpu().detach().numpy()
    
    
    sorted_softmax_values, lower_index, upper_index, mean_samples = credible_interval(softmax_values[:, :, 0].flatten(), 0.64)
 
    sorted_softmax_values_fr1 = sorted_softmax_values[lower_index:upper_index]
    sorted_softmax_values_fr2 = 1 - sorted_softmax_values[lower_index:upper_index]

    softmax_mean = np.vstack((mean_samples, 1-mean_samples)).T
    softmax_credible = np.vstack((sorted_softmax_values_fr1, sorted_softmax_values_fr2)).T
    
    entropy, mutual_info, entropy_singlepass = entropy_MI(softmax_credible, 
                                                          samples_iter= len(softmax_credible[:,0]))

    pred_mean = np.argmax(softmax_mean, axis = 1)
    error_mean = (pred_mean != target)*1


    pred = np.argmax(softmax_credible, axis = 1)

    y_test_all = np.tile(target, len(softmax_credible[:,0]))

    errors =  np.mean((pred != y_test_all).astype('uint8'))




    avg_error_mean.append(error_mean)
    error_all.append(errors)
    entropy_all.append(entropy/np.log(2))    
    mi_all.append(mutual_info/np.log(2))
    aleat_all.append(entropy_singlepass/np.log(2))


n_bins = 8
    
uce  = calibration(path, np.array(error_all), np.array(entropy_all), n_bins, x_label = 'predictive entropy')
print("Predictive Entropy")
print("uce = ", np.round(uce, 2))

uce  = calibration(path, np.array(error_all), np.array(mi_all), n_bins, x_label = 'mutual information')
print("Mutual Information")
print("uce = ", np.round(uce, 2))
 

uce  = calibration(path, np.array(error_all), np.array(aleat_all), n_bins, x_label = 'average entropy')
print("Average Entropy")
print("uce = ", np.round(uce, 2))

print((np.array(error_all)).mean())
print((np.array(error_all)).std())

print("Average of expected error")
print((np.array(avg_error_mean)).mean())
print((np.array(avg_error_mean)).std())

