""" Script for plotting energy scores"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import radiogalaxies_bnns.inference.utils as utils

paths = utils.Path_Handler()._dict()

"combine energy scores for all methods into a single csv"
# mlp = pd.read_csv(paths['project'] / 'results' / 'ood' / 'mlp_energy_scores.csv', index_col=0)
# dropout = pd.read_csv(paths['project'] / 'results' / 'ood' / 'dropout_energy_scores.csv', index_col=0)
# ensemble = pd.read_csv(paths['project'] / 'results' / 'ood' / 'ensembles_energy_scores.csv', index_col=0)
# lla = pd.read_csv('paths['project'] / 'results' / 'ood' / 'lla_energy_scores.csv', index_col=0)
# vi = pd.read_csv('paths['project'] / 'results' / 'ood' / 'vi_energy_scores.csv', index_col=0)
# hmc = pd.read_csv('paths['project'] / 'results' / 'ood' / 'hmc_energy_scores.csv', index_col=0)

# merged_df = pd.concat([mlp, dropout, ensemble, lla, vi, hmc], axis =1)
# merged_df.to_csv(paths['project'] / 'results' / 'ood' / 'energy_scores.csv')
 
energy_score_df = pd.read_csv(paths['project'] / 'results' / 'ood' / 'energy_scores.csv', index_col=0)


binwidth = 0.1
ylim_lower = 0
ylim_upper = 10

bin_lower = -30
bin_upper = 0.2
x_tick = 2 #gap for xticks

hmc_cols = ['HMC MB Conf', 'HMC Galaxy MNIST', 'HMC MIGHTEE']
vi_cols = ['VI MB Conf', 'VI Galaxy MNIST', 'VI MIGHTEE']
lla_cols = ['LLA MB Conf', 'LLA Galaxy MNIST', 'LLA MIGHTEE']
dropout_cols = ['Dropout MB Conf', 'Dropout Galaxy MNIST', 'Dropout MIGHTEE']
ensemble_cols = ['Ensemble MB Conf', 'Ensemble Galaxy MNIST', 'Ensemble MIGHTEE']
mlp_cols = ['MLP MB Conf', 'MLP Galaxy MNIST', 'MLP MIGHTEE']


plt.clf()
plt.figure(dpi=300)
sns.histplot(energy_score_df[hmc_cols], 
             binrange = [bin_lower, bin_upper], 
             binwidth = binwidth)
plt.ylim(ylim_lower, ylim_upper)
plt.xlabel('Energy Score')
plt.xticks(np.arange(bin_lower, bin_upper, x_tick))
plt.savefig(paths['project'] / 'results' / 'ood' / 'energy_hist_hmc1.png')

plt.clf()
plt.figure(dpi=300)
sns.histplot(energy_score_df[vi_cols], 
             binrange = [bin_lower, bin_upper], 
             binwidth = binwidth)
plt.ylim(ylim_lower, ylim_upper)
plt.xlabel('Energy Score')
plt.xticks(np.arange(bin_lower, bin_upper, x_tick))
plt.savefig(paths['project'] / 'results' / 'ood' / 'energy_hist_vi.png')

plt.clf()
plt.figure(dpi=300)
sns.histplot(energy_score_df[lla_cols], 
             binrange = [bin_lower, bin_upper], 
             binwidth = binwidth)
plt.ylim(ylim_lower, ylim_upper)
plt.xlabel('Energy Score')
plt.xticks(np.arange(bin_lower, bin_upper, x_tick))
plt.savefig(paths['project'] / 'results' / 'ood' / 'energy_hist_lla.png')


plt.clf()
plt.figure(dpi=300)
sns.histplot(energy_score_df[dropout_cols], 
             binrange = [bin_lower, bin_upper], 
             binwidth = binwidth)
plt.ylim(ylim_lower, ylim_upper)
plt.xlabel('Energy Score')
plt.xticks(np.arange(bin_lower, bin_upper, x_tick))
plt.savefig(paths['project'] / 'results' / 'ood' / 'energy_hist_dropout.png')

plt.clf()
plt.figure(dpi=300)
sns.histplot(energy_score_df[ensemble_cols], 
             binrange = [bin_lower, bin_upper], 
             binwidth = binwidth)
plt.ylim(ylim_lower, ylim_upper)
plt.xlabel('Energy Score')
plt.xticks(np.arange(bin_lower, bin_upper, x_tick))
plt.savefig(paths['project'] / 'results' / 'ood' / 'energy_hist_ensemble.png')

plt.clf()
plt.figure(dpi=300)
sns.histplot(energy_score_df[mlp_cols].rename(columns={"MLP MB Conf": "MAP MB Conf", 
"MLP Galaxy MNIST": "MAP Galaxy MNIST",
 'MLP MIGHTEE': 'MAP MIGHTEE'}), binrange = [bin_lower, bin_upper], 
             binwidth = binwidth)
plt.ylim(ylim_lower, ylim_upper)
plt.xlabel('Energy Score')
plt.xticks(np.arange(bin_lower, bin_upper, x_tick))
plt.savefig(paths['project'] / 'results' / 'ood' / 'energy_hist_mlp.png')

exit()