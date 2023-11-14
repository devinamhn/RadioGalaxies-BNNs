import torch
import hamiltorch

import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import InterpolationMode
from pytorch_lightning.demos.mnist_datamodule import MNIST
from torch.utils.data import DataLoader, random_split
from datamodules import MNISTDataModule, MiraBestDataModule, testloader_mb_uncert
from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl
import utils
from PIL import Image

from models import MLP, LeNet

from matplotlib import pyplot as plt
import mirabest
from uncertainty import entropy_MI, overlapping, GMM_logits, calibration
import csv


hamiltorch.set_random_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LeNet(1, 2) #MLP(150, 200, 10)

path = './results/temp/thin1000/' #'./results/checkpt/' #'./results/galahad/hamilt/testing/15000steps/'
params_hmc = torch.load(path+'thin_chain'+str(0), map_location = torch.device(device))
print(len(params_hmc))

tau_list = []
tau = 100. # 10.#./100. # 1/50

for w in model.parameters():
    tau_list.append(tau)
tau_list = torch.tensor(tau_list).to(device)

config_dict, config = utils.parse_config('config_mb.txt')

datamodule = MiraBestDataModule(config_dict, hmc=True)

test_loader, test_data1, data_type, test_data = testloader_mb_uncert(config_dict['output']['test_data'], config_dict['data']['datadir'])

for i, (x_test, y_test) in enumerate(test_loader):
    x_test, y_test = x_test.to(device), y_test.to(device)

print(len(y_test))




pred_list, log_prob_list = hamiltorch.predict_model(model, x = x_test, y = y_test, samples=params_hmc, model_loss= 'multi_class_linear_output', tau_out=1., tau_list=tau_list)
_, pred = torch.max(pred_list, 2)


# multi_class_log_softmax_output
# multi_class_linear_output

# acc = []
# acc = torch.zeros( int(len(params_hmc))-1)
# nll = torch.zeros( int(len(params_hmc))-1)
# ensemble_proba = F.softmax(pred_list[0], dim=-1)

# print(pred_list.shape)

# for s in range(1,len(params_hmc)):
#     _, pred = torch.max(pred_list[:s].mean(0), -1)
#     acc[s-1] = (pred.float() == y_test.flatten()).sum().float()/y_test.shape[0]

#     # print(pred_list[s][103])
#     # print('FRI', F.softmax(pred_list[s][103], dim=-1)[0], y_test[103])
#     # print('FRII', F.softmax(pred_list[s][103], dim=-1)[1])
#     # softmax_list.append((F.softmax(pred_list[s], dim = -1)).detach().numpy())
    
    
#     ensemble_proba += F.softmax(pred_list[s], dim=-1) #ensemble prob is being added - save all the softmax values
#     nll[s-1] = F.nll_loss(torch.log(ensemble_proba.cpu()/(s+1)), y_test[:].long().cpu().flatten(), reduction='mean')

# print("test accuracy", torch.mean(acc), torch.std(acc))

error_all = []
entropy_all = []
mi_all = []
aleat_all =[]
    
fr1 = 0
fr2 = 0

indices = np.arange(0, len(test_data), 1)
print(indices)

#for each sample in the test set:
for index in indices:

    # x_test = x_test[index] #torch.unsqueeze(x_test[index])
    # y_test = y_test[index] #torch.unsqueeze(x_test[index])
    x_test = torch.unsqueeze(torch.tensor(test_data[index][0]),0)
    y_test = torch.unsqueeze(torch.tensor(test_data[index][1]),0)

    # x_test[0]
    # print(x_test, y_test)

    # target = y_test[index].detach().numpy()
    target = y_test.detach().numpy().flatten()[0]

    if(target == 0):
        fr1+= 1
    elif(target==1):
        fr2+= 1

    pred_list, log_prob_list = hamiltorch.predict_model(model, x = x_test, y = y_test, samples=params_hmc, model_loss='multi_class_linear_output', tau_out=1., tau_list=tau_list)
    softmax_ = F.softmax(pred_list, dim=-1)

    pred = softmax_.mean(dim=1).argmax(dim=-1).numpy().flatten()
    pred1 = pred_list.mean(dim=1).argmax(dim=-1)

    y_test_all = np.tile(y_test.detach().numpy().flatten()[0], len(params_hmc))

    # print(pred)
    # print(pred1)
    # print(y_test_all)

    # print(pred != y_test_all)

    errors =  np.mean((pred != y_test_all).astype('uint8'))
    # print(errors)

    softmax = np.array(softmax_).reshape((len(params_hmc), 2))
    logits = pred #np.array(pred) 

    mean_logits = np.mean(logits,axis=0)
    var_logits = np.std(logits,axis=0)

    # print(softmax_list)

    # print(softmax[:,0])

    entropy, mutual_info, entropy_singlepass = entropy_MI(softmax, samples_iter= len(params_hmc))
    
    # print(entropy)
    # print(mutual_info)

    error_all.append(errors)
    entropy_all.append(entropy/np.log(2))    
    mi_all.append(mutual_info/np.log(2))


# print(error_all)
# print(entropy_all)




fr1_start = 0 #{0}
fr1_end = fr1 #{49, 68}
fr2_start = fr1 #49, 68
fr2_end = len(indices) #len(val_indices) #{104, 145}
    
print(fr1_start, fr1_end, fr2_start, fr2_end)
    
#print(error_all)
#entropy, mutual_info, entropy_singlepass = entropy_MI(softmax, samples_iter)
n_bins = 8
    
uce  = calibration(path, np.array(error_all), np.array(entropy_all), n_bins, x_label = 'predictive entropy')
print("Predictive Entropy")
print("uce = ", np.round(uce, 2))
uce_0  = calibration(path, np.array(error_all[fr1_start:fr1_end]), np.array(entropy_all[fr1_start:fr1_end]), n_bins, x_label = 'predictive entropy') 
print("UCE FRI= ", np.round(uce_0, 2))
uce_1  = calibration(path, np.array(error_all[fr2_start:fr2_end]), np.array(entropy_all[fr2_start:fr2_end]), n_bins, x_label = 'predictive entropy')
print("UCE FRII = ", np.round(uce_1, 2))
cUCE = (uce_0 + uce_1)/2 
print("cUCE=", np.round(cUCE, 2))
    
#max_mi = np.amax(np.array(mi_all))
#print("max mi",max_mi)
#mi_all = mi_all/max_mi
#print(mi_all)
#print("max mi",np.amax(mi_all))
uce  = calibration(path, np.array(error_all), np.array(mi_all), n_bins, x_label = 'mutual information')
print("Mutual Information")
print("uce = ", np.round(uce, 2))
uce_0  = calibration(path, np.array(error_all[fr1_start:fr1_end]), np.array(mi_all[fr1_start:fr1_end]), n_bins, x_label = 'mutual information') 
print("UCE FRI= ", np.round(uce_0, 2))
uce_1  = calibration(path, np.array(error_all[fr2_start:fr2_end]), np.array(mi_all[fr2_start:fr2_end]), n_bins, x_label = 'mutual information')
print("UCE FRII = ", np.round(uce_1, 2))
cUCE = (uce_0 + uce_1)/2 
print("cUCE=", np.round(cUCE,2))  

print("mean and std of error")
print(error_all)
print(np.mean(error_all)*100)
print(np.std(error_all))

# uce  = calibration(path, np.array(error_all), np.array(aleat_all), n_bins, x_label = 'average entropy')
# print("Average Entropy")
# print("uce = ", np.round(uce, 2))
# uce_0  = calibration(path, np.array(error_all[fr1_start:fr1_end]), np.array(aleat_all[fr1_start:fr1_end]), n_bins, x_label = 'average entropy') 
# print("UCE FRI= ",np.round(uce_0, 2))
# uce_1  = calibration(path, np.array(error_all[fr2_start:fr2_end]), np.array(aleat_all[fr2_start:fr2_end]), n_bins, x_label = 'average entropy')
# print("UCE FRII = ", np.round(uce_1, 2))
# cUCE = (uce_0 + uce_1)/2 
# print("cUCE=", np.round(cUCE, 2))  
    
   