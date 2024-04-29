import torch
import hamiltorch
from matplotlib import pyplot as plt
import csv
import seaborn as sns
import pandas as pd
import numpy as np

from radiogalaxies_bnns.inference.utils import Path_Handler

import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler
import torch.nn.functional as F
import torchvision

from pathlib import Path
from radiogalaxies_bnns.inference.models import LeNet, LeNetDrop
import radiogalaxies_bnns.inference.utils as utils
from radiogalaxies_bnns.inference.datamodules import MNISTDataModule, MiraBestDataModule, testloader_mb_uncert
from radiogalaxies_bnns.eval.uncertainty.uncertainty import entropy_MI, overlapping, GMM_logits, calibration

from pathlib import Path

from radiogalaxies_bnns.datasets import mirabest
from radiogalaxies_bnns.eval.uncertainty.uncertainty import entropy_MI, calibration
from galaxy_mnist import GalaxyMNIST, GalaxyMNISTHighrez

from cata2data import CataData
from torch.utils.data import DataLoader
from radiogalaxies_bnns.datasets.mightee import MighteeZoo

def energy_function(logits, T = 1):
    
    # print(-torch.logsumexp(pred_list_mbconf, dim = 2))
    mean_energy = torch.mean(-torch.logsumexp(logits, dim = 2), dim = 0).cpu().detach().numpy()
    std_energy = torch.std(-torch.logsumexp(logits, dim = 2), dim = 0).cpu().detach().numpy()

    return mean_energy


# energy_score_df = pd.read_csv('radiogalaxies_bnns/results/ood/hmc_energy_scores.csv', index_col=0)

# binwidth = 1#0.10
# ylim_lower = 0
# ylim_upper = 7

# bin_lower = -30 #-80
# bin_upper = 2

# plt.clf()
# plt.figure(dpi=300)
# sns.histplot(energy_score_df[['HMC MB Conf', 'HMC Galaxy MNIST', 'HMC MIGHTEE']], binrange = [bin_lower, bin_upper], 
#              binwidth = binwidth)
# plt.ylim(ylim_lower, ylim_upper)
# plt.xlabel('Negative Energy')#Negative
# plt.xticks(np.arange(bin_lower, bin_upper, 5))
# plt.savefig('radiogalaxies_bnns/results/ood/energy_hist_hmc.png')



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LeNet(1, 2)
path = '/share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/results/inits/'
params_hmc = torch.load(path+'/thin_chain'+str(0), map_location=torch.device(device)) 

burn_in = 0 #20
params_hmc = params_hmc[burn_in:]

tau_list = []
tau = 100. 

for w in model.parameters():
    tau_list.append(tau)
tau_list = torch.tensor(tau_list).to(device)

config_path = '/share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/inference/hmc/config_mb.txt'
config_dict, config = utils.parse_config(config_path)

datamodule = MiraBestDataModule(config_dict, hmc=True)
# train_loader = datamodule.train_dataloader()
# validation_loader = datamodule.val_dataloader()
# test_loader = datamodule.test_dataloader()

#MB Confident
test_loader, test_data1, data_type, test_data = testloader_mb_uncert(config_dict['output']['test_data'], config_dict['data']['datadir'])

for i, (x_test, y_test) in enumerate(test_loader):
    x_test, y_test = x_test.to(device), y_test.to(device)

#MB Uncertain

test_loader_mbuncert, test_data1_mbuncert, data_type_mbuncert, test_data_mbuncert = testloader_mb_uncert("MBFRUncertain", config_dict['data']['datadir'])

for i, (x_test_mbuncert, y_test_mbuncert) in enumerate(test_loader_mbuncert):
    x_test_mbuncert, y_test_mbuncert = x_test_mbuncert.to(device), y_test_mbuncert.to(device)


#MB Hybrid
test_loader_mbhybrid, test_data1_mbhybrid, data_type_mbhybrid, test_data_mbhybrid = testloader_mb_uncert("MBHybrid", config_dict['data']['datadir'])

for i, (x_test_mbhybrid, y_test_mbhybrid) in enumerate(test_loader_mbhybrid):
    x_test_mbhybrid, y_test_mbhybrid = x_test_mbhybrid.to(device), y_test_mbhybrid.to(device)


#Galaxy MNIST

transform = torchvision.transforms.Compose([ torchvision.transforms.ToTensor(),
torchvision.transforms.Resize((150,150), antialias = True), 
torchvision.transforms.Grayscale(),
])

# 64 pixel images
train_dataset = GalaxyMNISTHighrez(
    root='./dataGalaxyMNISTHighres',
    download=True,
    train=True,  # by default, or set False for test set
    transform = transform
)

test_dataset = GalaxyMNISTHighrez(
    root='./dataGalaxyMNISTHighres',
    download=True,
    train=False,  # by default, or set False for test set
    transform = transform
)

gal_mnist_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=104, shuffle = False)

for i, (x_test_galmnist, y_test_galmnist) in enumerate(gal_mnist_test_loader):
    x_test_galmnist, y_test_galmnist = x_test_galmnist.to(device), y_test_galmnist.to(device)
    y_test_galmnist = torch.zeros(104).to(device)
    if(i==0):
        break

transform = torchvision.transforms.Compose(
[
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(150),  # Rescale to adjust for resolution difference between MIGHTEE & RGZ - was 70
    torchvision.transforms.Normalize((1.59965605788234e-05,), (0.0038063037602458706,)),
]
)
paths = Path_Handler()._dict()
set = 'certain'

data = MighteeZoo(path=paths["mightee"], transform=transform, set="certain")
test_loader = DataLoader(data, batch_size=len(data))
for i, (x_test_mightee, y_test_mightee) in enumerate(test_loader):
    x_test_mightee, y_test_mightee = x_test_mightee.to(device), y_test_mightee.to(device)
        
test_data = data

temp = 1

pred_list_mbconf, log_prob_list_mbconf = hamiltorch.predict_model(model, x = x_test, y = y_test, 
                                            samples=params_hmc, model_loss='multi_class_linear_output', 
                                            tau_out=1., tau_list=tau_list)#, temp = temp)

pred_list_mbuncert, log_prob_list_mbuncert = hamiltorch.predict_model(model, x = x_test_mbuncert, y = y_test_mbuncert, 
                                            samples=params_hmc, model_loss='multi_class_linear_output', 
                                            tau_out=1., tau_list=tau_list)#, temp = temp)

pred_list_mbhybrid, log_prob_list_mbhybrid = hamiltorch.predict_model(model, x = x_test_mbhybrid, y = y_test_mbhybrid, 
                                            samples=params_hmc, model_loss='multi_class_linear_output', 
                                            tau_out=1., tau_list=tau_list)#, temp = temp)

pred_list_gal_mnist, log_prob_list_gal_mnist = hamiltorch.predict_model(model, x = x_test_galmnist, y = y_test_galmnist, 
                                                    samples=params_hmc, model_loss='multi_class_linear_output', 
                                                    tau_out=1., tau_list=tau_list)#, temp = temp)


pred_list_mightee, log_prob_list_mightee = hamiltorch.predict_model(model, x = x_test_mightee, y = y_test_mightee, 
                                                    samples=params_hmc, model_loss='multi_class_linear_output', 
                                                    tau_out=1., tau_list=tau_list)#, temp = temp)



mean_energy_conf = energy_function(pred_list_mbconf)
mean_energy_galmnist = energy_function(pred_list_gal_mnist)
mean_energy_uncert = energy_function(pred_list_mbuncert)
mean_energy_hybrid = energy_function(pred_list_mbhybrid)
mean_energy_mightee = energy_function(pred_list_mightee)


s1 = pd.Series(mean_energy_conf, name = 'HMC MB Conf')
s2 = pd.Series(mean_energy_uncert, name = 'HMC MB Uncert')
s3 = pd.Series(mean_energy_hybrid, name = 'HMC MB Hybrid')
s4 = pd.Series(mean_energy_galmnist, name = 'HMC Galaxy MNIST')
s5 = pd.Series(mean_energy_mightee, name = 'HMC MIGHTEE')
energy_score_df = pd.DataFrame([s1, s2, s3, s4, s5]).T



energy_score_df.to_csv('radiogalaxies_bnns/results/ood/hmc_energy_scores.csv')

# energy_score_df = pd.read_csv('radiogalaxies_bnns/results/ood/hmc_energy_scores.csv', index_col=0)

# binwidth = 0.1#0.10
# ylim_lower = 0
# ylim_upper = 7

# bin_lower = -80
# bin_upper = 10

# plt.clf()
# plt.figure(dpi=300)
# sns.histplot(energy_score_df[['HMC MB Conf', 'HMC Galaxy MNIST', 'HMC MIGHTEE']], binrange = [bin_lower, bin_upper], 
#              binwidth = binwidth)
# plt.ylim(ylim_lower, ylim_upper)
# plt.xlabel('Negative Energy')#Negative
# plt.xticks(np.arange(-80, 10, 10))
# plt.savefig('radiogalaxies_bnns/results/ood/energy_hist_hmc.png')


exit()

