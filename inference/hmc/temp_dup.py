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
from datamodules import MNISTDataModule, MiraBestDataModule
from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl
import utils
from PIL import Image

from models import MLP, LeNet
import sys

config_dict, config = utils.parse_config('config_mb.txt')
jobid = int(sys.argv[1])
print(jobid)

hamiltorch.set_random_seed(config_dict['training']['seed']) #+ jobid)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model = MLP(28, 200, 10)
model = LeNet(1, 2, jobid)

datamodule = MiraBestDataModule(config_dict, hmc=True)
train_loader = datamodule.train_dataloader()
validation_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()

# datamodule = MNISTDataModule(128, hmc=True)
# train_loader = datamodule.train_dataloader()
# validation_loader = datamodule.val_dataloader()
# test_loader = datamodule.test_dataloader()

step_size = 0.0001 #0.01# 0.003#0.002
num_samples = 200000 #2000 # 3000
L = 50 #3
tau_out = 1.
normalizing_const = 1.
burn = 0 #GPU: 3000


tau_list = []
tau = 100. # 10.#./100. # 1/50
for w in model.parameters():
#     print(w.nelement())
#     tau_list.append(tau/w.nelement())
    tau_list.append(tau)
tau_list = torch.tensor(tau_list).to(device)

# tau = 1 => std =1
# tau = 10 => std =0.31
# tau = 100 => std =0.1
# tau = 200 => std =0.07
# tau = 500 => std =0.04
# tau = 1000 => std =0.03
# tau = 10000 => std = 0.01 

params_init = hamiltorch.util.flatten(model).to(device).clone()

batch = 1
num_batches = 1
for batch, (x_train, y_train) in enumerate(train_loader):
    x_train, y_train = x_train.to(device), y_train.to(device)

checkpt = 1000
n_checkpt = int(num_samples/checkpt)

path = './results/inits/mb_out_' + str(jobid) + '/'

for i in range(n_checkpt):
    if(i!=0):
        #load in the previous checkpoint
        params_init = torch.load(path + 'params_hmc_checkpt.pt')
        #burnin
    params_hmc = hamiltorch.sample_model(model, x_train, y_train, params_init=params_init, model_loss='multi_class_linear_output', num_samples=checkpt, burn = burn,
                               step_size=step_size, num_steps_per_sample=L,tau_out=tau_out, tau_list=tau_list, normalizing_const=normalizing_const)
    
    torch.save(params_hmc, path + 'params_hmc_'+ str(i) + '.pt')
    torch.save(params_hmc[checkpt-1][:], path + 'params_hmc_checkpt.pt')

    #print(params_hmc[checkpt-1][:])

exit()
# ###predictions###

# path = './results/temp/mb_out_' + str(jobid) + '/'

# for i in range(n_checkpt)
#     params_hmc = torch.load(path+'')


# path = './results/10000steps/'

# params_hmc = torch.load(path + 'params_hmc.pt')

for i, (x_test, y_test) in enumerate(test_loader):
    x_test, y_test = x_test.to(device), y_test.to(device)

pred_list, log_prob_list = hamiltorch.predict_model(model, x = x_test, y = y_test, samples=params_hmc, model_loss='multi_class_linear_output', tau_out=1., tau_list=tau_list)
_, pred = torch.max(pred_list, 2)
acc = []
acc = torch.zeros( int(len(params_hmc))-1)
nll = torch.zeros( int(len(params_hmc))-1)
ensemble_proba = F.softmax(pred_list[0], dim=-1)


for s in range(1,len(params_hmc)):
    _, pred = torch.max(pred_list[:s].mean(0), -1)
    acc[s-1] = (pred.float() == y_test.flatten()).sum().float()/y_test.shape[0]
    ensemble_proba += F.softmax(pred_list[s], dim=-1) #ensemble orob is being added - save all the softmax values
    nll[s-1] = F.nll_loss(torch.log(ensemble_proba.cpu()/(s+1)), y_test[:].long().cpu().flatten(), reduction='mean')

torch.save(ensemble_proba, path+"ensemble_prob.pt")
# torch.save(params_hmc,'params_hmc.pt')
# torch.save(acc,'acc_train.pt')
# torch.save(nll,'nll_train.pt')

print("test accuracy", torch.mean(acc), torch.std(acc))
#print("ensemble probability", ensemble_proba)


