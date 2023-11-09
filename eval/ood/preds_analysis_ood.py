import torch
import sys

sys.path.append('/share/nas2/dmohan/mcmc/hamildev/hamiltorch')
import hamiltorch

import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import numpy as np
import torchvision 
# import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import InterpolationMode
from pytorch_lightning.demos.mnist_datamodule import MNIST
from torch.utils.data import DataLoader, random_split
from datamodules import MNISTDataModule, MiraBestDataModule, testloader_mb_uncert
from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl
import utils
from PIL import Image
from galaxy_mnist import GalaxyMNIST, GalaxyMNISTHighrez

from models import MLP, LeNet

from matplotlib import pyplot as plt
import mirabest
from uncertainty import entropy_MI, overlapping, GMM_logits, calibration
import csv
import seaborn as sns
import pandas as pd

def energy_function(logits, T = 1):
    
    # print(-torch.logsumexp(pred_list_mbconf, dim = 2))
    mean_energy = torch.mean(-torch.logsumexp(logits, dim = 2), dim = 0).detach().numpy()
    std_energy = torch.std(-torch.logsumexp(logits, dim = 2), dim = 0).detach().numpy()

    return mean_energy



energy_score_df = pd.read_csv('./results/ood/energy_scores.csv', index_col=0)

energy_score_exp_df = np.exp(energy_score_df)


g = sns.PairGrid(energy_score_df.drop(columns = ['VI MB Conf', 'HMC MB Conf', 
                                                 'Dropout MB Conf', 'LLA MB Conf', 
                                                 'CNN MB Conf'] ), diag_sharey=False)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.map_diag(sns.histplot)
plt.savefig('./energy_pairgrid_ood.png')


plt.clf()
g = sns.PairGrid(energy_score_df.drop(columns =  ['VI Galaxy MNIST', 'HMC Galaxy MNIST ', 
                                             'Dropout Galaxy MNIST', 'LLA Galaxy MNIST', 
                                             'CNN Galaxy MNIST'] ), diag_sharey=False)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.map_diag(sns.histplot)
plt.savefig('./energy_pairgrid_id.png')





# sns.pairplot(energy_score_exp_df.drop(columns = ['VI MB Conf', 'HMC MB Conf', 
#                                              'Dropout MB Conf', 'LLA MB Conf', 'CNN MB Conf'] ))
# # plt.xlabel(' Negative Energy')#Negative
# plt.savefig('./energy_exp_pairplot_ood.png')

# plt.clf()
# sns.pairplot(energy_score_exp_df.drop(columns = ['VI Galaxy MNIST', 'HMC Galaxy MNIST ', 
#                                              'Dropout Galaxy MNIST', 'LLA Galaxy MNIST', 
#                                              'CNN Galaxy MNIST'] ))
# # plt.xlabel(' Negative Energy')#Negative
# plt.savefig('./energy_exp_pairplot_id.png')


exit()
######## HISTOGRAMS for exp of energy scores
binwidth = 0.05
ylim_lower = 0
ylim_upper = 10

bin_lower = 0
bin_upper = 2

plt.clf()
sns.histplot(energy_score_exp_df[['CNN MB Conf', 'CNN Galaxy MNIST']], binrange = [bin_lower, bin_upper], binwidth = binwidth)
plt.ylim(ylim_lower, ylim_upper)
plt.xlabel(' Exp (Negative Energy)')#Negative
plt.xticks(np.arange(bin_lower, bin_upper, 1))
plt.savefig('./energy_exp_hist_cnn.png')

plt.clf()
sns.histplot(energy_score_exp_df[['VI MB Conf', 'VI Galaxy MNIST']], binrange = [bin_lower, bin_upper], binwidth = binwidth)
plt.ylim(ylim_lower, ylim_upper)
plt.xlabel(' Exp (Negative Energy)')#Negative
plt.xticks(np.arange(bin_lower, bin_upper, 1))
plt.savefig('./energy_exp_hist_vi.png')


plt.clf()
sns.histplot(energy_score_exp_df[['Dropout MB Conf', 'Dropout Galaxy MNIST']], binrange = [bin_lower, bin_upper], binwidth = binwidth)
plt.ylim(ylim_lower, ylim_upper)
plt.xlabel(' Exp (Negative Energy)')#Negative
plt.xticks(np.arange(bin_lower, bin_upper, 1))
plt.savefig('./energy_exp_hist_dropout.png')

plt.clf()
sns.histplot(energy_score_exp_df[['HMC MB Conf', 'HMC Galaxy MNIST ']], binrange = [bin_lower, bin_upper], binwidth = binwidth)
plt.ylim(ylim_lower, ylim_upper)
plt.xlabel(' Exp (Negative Energy)')#Negative
plt.xticks(np.arange(bin_lower, bin_upper, 1))
plt.savefig('./energy_exp_hist_hmc.png')

plt.clf()
sns.histplot(energy_score_exp_df[['LLA MB Conf', 'LLA Galaxy MNIST']], binrange = [bin_lower, bin_upper], binwidth = binwidth)
plt.ylim(ylim_lower, ylim_upper)
plt.xlabel(' Exp (Negative Energy)')#Negative
plt.xticks(np.arange(bin_lower, bin_upper, 1))
plt.savefig('./energy_exp_hist_lla.png')



exit()
######## HISTOGRAMS for energy scores
binwidth = 0.10
ylim_lower = 0
ylim_upper = 10

bin_lower = 1
bin_upper = 4


plt.clf()
sns.histplot(energy_score_df[['CNN MB Conf', 'CNN Galaxy MNIST']], binrange = [-8, 1], binwidth = binwidth)
plt.ylim(ylim_lower, ylim_upper)
plt.xlabel(' Negative Energy')#Negative
plt.xticks(np.arange(-8, 2, 1))
plt.savefig('./energy_hist_cnn.png')

plt.clf()
sns.histplot(energy_score_df[['VI MB Conf', 'VI Galaxy MNIST']], binrange = [-8, 1], binwidth = binwidth)
plt.ylim(ylim_lower, ylim_upper)
plt.xlabel(' Negative Energy')#Negative
plt.xticks(np.arange(-8, 2, 1))
plt.savefig('./energy_hist_vi.png')


plt.clf()
sns.histplot(energy_score_df[['Dropout MB Conf', 'Dropout Galaxy MNIST']], binrange = [-8, 1], binwidth = binwidth)
plt.ylim(ylim_lower, ylim_upper)
plt.xlabel(' Negative Energy')#Negative
plt.xticks(np.arange(-8, 2, 1))
plt.savefig('./energy_hist_dropout.png')

plt.clf()
sns.histplot(energy_score_df[['HMC MB Conf', 'HMC Galaxy MNIST ']], binrange = [-8, 1], binwidth = binwidth)
plt.ylim(ylim_lower, ylim_upper)
plt.xlabel(' Negative Energy')#Negative
plt.xticks(np.arange(-8, 2, 1))
plt.savefig('./energy_hist_hmc.png')

plt.clf()
sns.histplot(energy_score_df[['LLA MB Conf', 'LLA Galaxy MNIST']], binrange = [-8, 1], binwidth = binwidth)
plt.ylim(ylim_lower, ylim_upper)
plt.xlabel(' Negative Energy')#Negative
plt.xticks(np.arange(-8, 2, 1))
plt.savefig('./energy_hist_lla.png')

# plt.clf()
# sns.histplot(energy_score_df.drop(columns = ['VI MB Conf', 'HMC MB Conf', 'Dropout MB Conf', 'LLA MB Conf', 
#                                              'CNN MB Conf']), binrange = [-2, 2], binwidth =0.5)
# plt.xlabel(' Negative Energy')#Negative
# plt.savefig('./energy_hist_ood.png')

# plt.clf()
# sns.histplot(energy_score_df.drop(columns = ['VI Galaxy MNIST', 'HMC Galaxy MNIST ', 'Dropout Galaxy MNIST', 
#                                              'LLA Galaxy MNIST', 'CNN Galaxy MNIST'] ), binrange = [-8, 1], binwidth = 0.5 )
# plt.xlabel(' Negative Energy')#Negative1
# plt.savefig('./energy_hist_id.png')


######## KDE PLOTS ##########


# plt.clf()
# print(energy_score_df)
# sns.kdeplot(energy_score_df.drop(columns = ['VI MB Conf', 'HMC MB Conf', 'Dropout MB Conf', 'LLA MB Conf'] ))
# plt.xlabel(' Negative Energy')#Negative
# plt.savefig('./energy_kde_ood.png')

# plt.clf()
# sns.kdeplot(energy_score_df.drop(columns = ['VI Galaxy MNIST', 'HMC Galaxy MNIST ', 'Dropout Galaxy MNIST', 'LLA Galaxy MNIST'] ))
# plt.xlabel(' Negative Energy')#Negative
# plt.savefig('./energy_kde_id.png')


# plt.clf()
# sns.kdeplot(energy_score_df[['VI MB Conf', 'VI Galaxy MNIST']])
# plt.xlabel(' Negative Energy')#Negative
# plt.savefig('./energy_kde_vi.png')


# plt.clf()
# sns.kdeplot(energy_score_df[['Dropout MB Conf', 'Dropout Galaxy MNIST']])
# plt.xlabel(' Negative Energy')#Negative
# plt.savefig('./energy_kde_dropout.png')

# plt.clf()
# sns.kdeplot(energy_score_df[['HMC MB Conf', 'HMC Galaxy MNIST ']])
# plt.xlabel(' Negative Energy')#Negative
# plt.savefig('./energy_kde_hmc.png')


# plt.clf()
# sns.kdeplot(energy_score_df[['LLA MB Conf', 'LLA Galaxy MNIST']])
# plt.xlabel(' Negative Energy')#Negative
# plt.savefig('./energy_kde_lla.png')


exit()

#
# hamiltorch.set_random_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LeNet(1, 2)

path = './results/temp/thin1000/' 

params_hmc = torch.load(path+'/thin_chain'+str(0), map_location=torch.device(device)) 
burn_in = 20
params_hmc = params_hmc[burn_in:]

tau_list = []
tau = 100. 

for w in model.parameters():
    tau_list.append(tau)
tau_list = torch.tensor(tau_list).to(device)

config_dict, config = utils.parse_config('config_mb.txt')

datamodule = MiraBestDataModule(config_dict, hmc=True)
# train_loader = datamodule.train_dataloader()
# validation_loader = datamodule.val_dataloader()
# test_loader = datamodule.test_dataloader()

#MB Confident
test_loader, test_data1, data_type, test_data = testloader_mb_uncert(config_dict['output']['test_data'], config_dict['data']['datadir'])

for i, (x_test, y_test) in enumerate(test_loader):
    x_test, y_test = x_test.to(device), y_test.to(device)
'''
#MB Uncertain

test_loader_mbuncert, test_data1_mbuncert, data_type_mbuncert, test_data_mbuncert = testloader_mb_uncert("MBFRUncertain", config_dict['data']['datadir'])

for i, (x_test_mbuncert, y_test_mbuncert) in enumerate(test_loader_mbuncert):
    x_test_mbuncert, y_test_mbuncert = x_test_mbuncert.to(device), y_test_mbuncert.to(device)


#MB Hybrid
test_loader_mbhybrid, test_data1_mbhybrid, data_type_mbhybrid, test_data_mbhybrid = testloader_mb_uncert("MBHybrid", config_dict['data']['datadir'])

for i, (x_test_mbhybrid, y_test_mbhybrid) in enumerate(test_loader_mbhybrid):
    x_test_mbhybrid, y_test_mbhybrid = x_test_mbhybrid.to(device), y_test_mbhybrid.to(device)

'''
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

temp = 1

pred_list_mbconf, log_prob_list_mbconf = hamiltorch.predict_model(model, x = x_test, y = y_test, 
                                            samples=params_hmc, model_loss='multi_class_linear_output', 
                                            tau_out=1., tau_list=tau_list, temp = temp)

# pred_list_mbuncert, log_prob_list_mbuncert = hamiltorch.predict_model(model, x = x_test_mbuncert, y = y_test_mbuncert, 
#                                             samples=params_hmc, model_loss='multi_class_linear_output', 
#                                             tau_out=1., tau_list=tau_list, temp = temp)

# pred_list_mbhybrid, log_prob_list_mbhybrid = hamiltorch.predict_model(model, x = x_test_mbhybrid, y = y_test_mbhybrid, 
#                                             samples=params_hmc, model_loss='multi_class_linear_output', 
#                                             tau_out=1., tau_list=tau_list, temp = temp)

pred_list_gal_mnist, log_prob_list_gal_mnist = hamiltorch.predict_model(model, x = x_test_galmnist, y = y_test_galmnist, 
                                                    samples=params_hmc, model_loss='multi_class_linear_output', 
                                                    tau_out=1., tau_list=tau_list, temp = temp)


print(pred_list_mbconf.shape)
# print(pred_list_mbuncert.shape)
# print(pred_list_mbhybrid.shape)
# print(pred_list_gal_mnist.shape)
print(torch.mean(pred_list_mbconf[:, :, :1], dim = 0).shape)
print(torch.mean(pred_list_mbconf[:, :, 1:], dim = 0).shape)
plt.scatter(torch.mean(pred_list_mbconf[:, :, :1], dim = 0), torch.mean(pred_list_mbconf[:, :, 1:], dim = 0), s = 1, label = 'MB Conf')
# plt.scatter(torch.mean(pred_list_gal_mnist[:, :, :1], dim = 0), torch.mean(pred_list_gal_mnist[:, :, 1:], dim = 0), s =1,
#              label = 'Galaxy MNIST')

# plt.scatter(pred_list_mbconf[:, :, :1], pred_list_mbconf[:, :, 1:], s = 1, label = 'MB Conf')
# plt.scatter(pred_list_mbuncert[:, :, :1], pred_list_mbuncert[:, :, 1:], s = 1, label = 'MB Uncert')
# plt.scatter(pred_list_mbhybrid[:, :, :1], pred_list_mbhybrid[:, :, 1:], s = 1, label = 'MB Hybrid')
# plt.scatter(pred_list_gal_mnist[:, :, :1], pred_list_gal_mnist[:, :, 1:], s =1,
#             label = 'Galaxy MNIST')
plt.legend()
plt.xlabel('Class 0 Logits')
plt.ylabel('class 1 Logits')
plt.savefig('./logits_mean_mbconf.png')

exit()
mean_energy_conf = energy_function(pred_list_mbconf)
mean_energy_galmnist =energy_function(pred_list_gal_mnist)
mean_energy_uncert = energy_function(pred_list_mbuncert)
mean_energy_hybrid = energy_function(pred_list_mbhybrid)

energy_score_df = pd.DataFrame({'MB Conf': mean_energy_conf, 
                                'Galaxy MNIST': mean_energy_galmnist,
                                }) #, mean_enerygy_galmnist.reshape(100, 1)])
# energy_score_df['dataset'] = dataset
print(energy_score_df)
energy_score_df.to_csv('./results/ood/hmc_energy_scores.csv')

sns.kdeplot(energy_score_df) #, hue = 'dataset')
plt.xlabel('Negative Energy')
plt.savefig('./enerygy_kde.png')



# plt.scatter(pred_list[:, :, :1], pred_list[:, :, 1:], s = 1, label = 'MB Conf')
# plt.scatter(pred_list_mbuncert[:, :, :1], pred_list_mbuncert[:, :, 1:], s = 1, label = 'MB Uncert')
# plt.scatter(pred_list_mbhybrid[:, :, :1], pred_list_mbhybrid[:, :, 1:], s = 1, label = 'MB Hybrid')
# plt.scatter(pred_list_gal_mnist[:, :, :1], pred_list_gal_mnist[:, :, 1:], s =1,
#              label = 'Galaxy MNIST')
# plt.legend()
# plt.xlabel('Class 0 Logits')
# plt.ylabel('class 1 Logits')
# plt.savefig('./logits.png')

exit()

