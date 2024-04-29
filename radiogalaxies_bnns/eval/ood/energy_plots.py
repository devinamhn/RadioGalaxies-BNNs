import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
import scipy
import pandas as pd
import seaborn as sns


# mlp = pd.read_csv('radiogalaxies_bnns/results/ood/mlp_energy_scores.csv', index_col=0)
# dropout = pd.read_csv('radiogalaxies_bnns/results/ood/dropout_energy_scores.csv', index_col=0)
# ensemble = pd.read_csv('radiogalaxies_bnns/results/ood/ensembles_energy_scores.csv', index_col=0)
# lla = pd.read_csv('radiogalaxies_bnns/results/ood/lla_energy_scores.csv', index_col=0)
# vi = pd.read_csv('radiogalaxies_bnns/results/ood/vi_energy_scores.csv', index_col=0)
# hmc = pd.read_csv('radiogalaxies_bnns/results/ood/hmc_energy_scores.csv', index_col=0)

# merged_df = pd.concat([mlp, dropout, ensemble, lla, vi, hmc], axis =1)
# merged_df.to_csv('radiogalaxies_bnns/results/ood/energy_scores.csv')

energy_score_df = pd.read_csv('radiogalaxies_bnns/results/ood/energy_scores.csv', index_col=0)

print(energy_score_df)


binwidth = 0.1 #0.1
ylim_lower = 0
ylim_upper = 10

bin_lower = -30# -30
bin_upper = 0.2 #-0.5
x_tick = 2 #2 #gap for xticks

hmc_cols = ['HMC MB Conf', 'HMC Galaxy MNIST', 'HMC MIGHTEE']#, 'HMC MB Uncert', 'HMC MB Hybrid']
vi_cols = ['VI MB Conf', 'VI Galaxy MNIST', 'VI MIGHTEE']#, 'VI MB Uncert', 'VI MB Hybrid']
lla_cols = ['LLA MB Conf', 'LLA Galaxy MNIST', 'LLA MIGHTEE']#, 'LLA MB Uncert', 'LLA MB Hybrid']
dropout_cols = ['Dropout MB Conf', 'Dropout Galaxy MNIST', 'Dropout MIGHTEE']#, 'Dropout MB Uncert', 'Dropout MB Hybrid']
ensemble_cols = ['Ensemble MB Conf', 'Ensemble Galaxy MNIST', 'Ensemble MIGHTEE']#, 'Ensemble MB Uncert', 'Ensemble MB Hybrid']
mlp_cols = ['MLP MB Conf', 'MLP Galaxy MNIST', 'MLP MIGHTEE']#, 'MLP MB Uncert', 'MLP MB Hybrid']

# hmc_cols = ['HMC MB Conf', 'HMC Galaxy MNIST']#, 'HMC MIGHTEE', 'HMC MB Uncert', 'HMC MB Hybrid']
# vi_cols = ['VI MB Conf', 'VI Galaxy MNIST']#, 'VI MIGHTEE', 'VI MB Uncert', 'VI MB Hybrid']
# lla_cols = ['LLA MB Conf', 'LLA Galaxy MNIST']#, 'LLA MIGHTEE', 'LLA MB Uncert', 'LLA MB Hybrid']
# dropout_cols = ['Dropout MB Conf', 'Dropout Galaxy MNIST']#, 'Dropout MIGHTEE', 'Dropout MB Uncert', 'Dropout MB Hybrid']
# ensemble_cols = ['Ensemble MB Conf', 'Ensemble Galaxy MNIST']#, 'Ensemble MIGHTEE', 'Ensemble MB Uncert', 'Ensemble MB Hybrid']
# mlp_cols = ['MLP MB Conf', 'MLP Galaxy MNIST']#, 'MLP MIGHTEE', 'MLP MB Uncert', 'MLP MB Hybrid']

# hmc_cols = ['HMC MB Conf', 'HMC MIGHTEE']#, 'HMC MB Uncert', 'HMC MB Hybrid']
# vi_cols = ['VI MB Conf', 'VI MIGHTEE'] #, 'VI MB Uncert', 'VI MB Hybrid']
# lla_cols = ['LLA MB Conf', 'LLA MIGHTEE'] #, 'LLA MB Uncert', 'LLA MB Hybrid']
# dropout_cols = ['Dropout MB Conf', 'Dropout MIGHTEE' ]#, 'Dropout MB Uncert', 'Dropout MB Hybrid']
# ensemble_cols = ['Ensemble MB Conf', 'Ensemble MIGHTEE']#, 'Ensemble MB Uncert', 'Ensemble MB Hybrid']
# mlp_cols = ['MLP MB Conf', 'MLP MIGHTEE'] #, 'MLP MB Uncert', 'MLP MB Hybrid']


plt.clf()
plt.figure(dpi=300)
sns.histplot(energy_score_df[hmc_cols], 
             binrange = [bin_lower, bin_upper], 
             binwidth = binwidth)
plt.ylim(ylim_lower, ylim_upper)
plt.xlabel('Energy Score')#Negative
plt.xticks(np.arange(bin_lower, bin_upper, x_tick))
plt.savefig('radiogalaxies_bnns/results/ood/energy_hist_hmc.png')


plt.clf()
plt.figure(dpi=300)
sns.histplot(energy_score_df[vi_cols], 
             binrange = [bin_lower, bin_upper], 
             binwidth = binwidth)
plt.ylim(ylim_lower, ylim_upper)
plt.xlabel('Energy Score')#Negative
plt.xticks(np.arange(bin_lower, bin_upper, x_tick))
plt.savefig('radiogalaxies_bnns/results/ood/energy_hist_vi.png')

plt.clf()
plt.figure(dpi=300)
sns.histplot(energy_score_df[lla_cols], 
             binrange = [bin_lower, bin_upper], 
             binwidth = binwidth)
plt.ylim(ylim_lower, ylim_upper)
plt.xlabel('Energy Score')#Negative
plt.xticks(np.arange(bin_lower, bin_upper, x_tick))
plt.savefig('radiogalaxies_bnns/results/ood/energy_hist_lla.png')


plt.clf()
plt.figure(dpi=300)
sns.histplot(energy_score_df[dropout_cols], 
             binrange = [bin_lower, bin_upper], 
             binwidth = binwidth)
plt.ylim(ylim_lower, ylim_upper)
plt.xlabel('Energy Score')#Negative
plt.xticks(np.arange(bin_lower, bin_upper, x_tick))
plt.savefig('radiogalaxies_bnns/results/ood/energy_hist_dropout.png')

plt.clf()
plt.figure(dpi=300)
sns.histplot(energy_score_df[ensemble_cols], 
             binrange = [bin_lower, bin_upper], 
             binwidth = binwidth)
plt.ylim(ylim_lower, ylim_upper)
plt.xlabel('Energy Score')#Negative
plt.xticks(np.arange(bin_lower, bin_upper, x_tick))
plt.savefig('radiogalaxies_bnns/results/ood/energy_hist_ensemble.png')

plt.clf()
plt.figure(dpi=300)
sns.histplot(energy_score_df[mlp_cols].rename(columns={"MLP MB Conf": "MAP MB Conf", 
"MLP Galaxy MNIST": "MAP Galaxy MNIST",
 'MLP MIGHTEE': 'MAP MIGHTEE'}), binrange = [bin_lower, bin_upper], 
             binwidth = binwidth)
plt.ylim(ylim_lower, ylim_upper)
plt.xlabel('Energy Score')#Negative
plt.xticks(np.arange(bin_lower, bin_upper, x_tick))
plt.savefig('radiogalaxies_bnns/results/ood/energy_hist_mlp.png')

exit()
i=1
plt.clf()
plt.figure(dpi=300)
sns.histplot(energy_score_df[[hmc_cols[i], vi_cols[i], lla_cols[i], dropout_cols[i], ensemble_cols[i], mlp_cols[i]]], 
             binrange = [bin_lower, bin_upper], 
             binwidth = binwidth)
plt.ylim(ylim_lower, ylim_upper)
plt.xlabel('Negative Energy')#Negative
plt.xticks(np.arange(bin_lower, bin_upper, x_tick))
plt.savefig('radiogalaxies_bnns/results/ood/energy_hist_galmnist.png')
exit()