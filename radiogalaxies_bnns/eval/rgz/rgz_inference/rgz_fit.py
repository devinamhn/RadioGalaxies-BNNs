from radiogalaxies_bnns.eval.rgz.rgz_inference.rgz_datamodules import RGZ_DataModule
from radiogalaxies_bnns.inference.utils import Path_Handler
from radiogalaxies_bnns.datasets.rgz108k import RGZ108k 
from torch.utils.data import DataLoader
import torchvision.transforms as T
from matplotlib import pyplot as plt 
import torch

import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats

# Get paths
paths = Path_Handler()._dict()

#Get energy scores
energy_score_df = pd.read_csv('radiogalaxies_bnns/results/ood/hmc_energy_scores_rgz.csv', index_col=0)


mean_energies = energy_score_df['HMC RGZ']
lognorm_fit = stats.lognorm.fit(-mean_energies, floc = 0)
print(f'lognormal fit: shape {lognorm_fit[0]}, loc {lognorm_fit[1]}, scale {lognorm_fit[2]}')
print(f'normal dist mean: {np.log(lognorm_fit[2])}, std: {lognorm_fit[0]}')

#check fit manually
# mu_hat = np.sum(np.log(-mean_energies))/len(mean_energies)
# var_hat = np.sum((np.log(-mean_energies) - mu_hat)**2)/len(mean_energies)
# std_hat = np.sqrt(var_hat)
# print(lognorm_fit[0], np.log(lognorm_fit[2]))
# print(mu_hat, var_hat, std_hat)

# Generate a range of values for plotting the fitted distribution
x = np.linspace(-mean_energies.min(), -mean_energies.max(), len(mean_energies))
pdf = stats.lognorm.pdf(x, *lognorm_fit)

# Plot the histogram of the data
plt.clf()
plt.figure(dpi=300)
sns.histplot(-mean_energies, kde=False, stat='density', binwidth=0.1, label='Mean energy')

# Plot the fitted lognormal distribution
plt.plot(x, pdf, 'r-', label='Fitted Lognormal Distribution')
plt.xlabel('Energy')
plt.ylabel('Density')
plt.legend()
plt.savefig('radiogalaxies_bnns/results/ood/energy_hist_hmc_rgz_fit_lognorm.png')

# normal fit to log of the values should return same parameters for the normal distribution 
norm_fit = stats.norm.fit(np.log(-mean_energies))

print(f' normal fit: {norm_fit}')
print(norm_fit[0], norm_fit[1])
print('Manual', np.log(-mean_energies).mean(), np.log(-mean_energies).std())

# Generate a range of values for plotting the fitted distribution
x = np.linspace(np.log(-mean_energies).min(), np.log(-mean_energies).max(), len(mean_energies))
pdf = stats.norm.pdf(x, *norm_fit)

# Plot the histogram of the data
plt.clf()
plt.figure(dpi=300)
sns.histplot(np.log(-mean_energies), kde=False, stat='density', binwidth=0.1, label='Mean energy')

# Plot the fitted lognormal distribution
plt.plot(x, pdf, 'r-', label='Fitted Normal Distribution')
plt.xlabel('Log Energy')
plt.ylabel('Density')
plt.legend()
plt.savefig('radiogalaxies_bnns/results/ood/energy_hist_hmc_rgz_normfit.png')

#get values from the actual fit

mean_norm= norm_fit[0] #np.log(-mean_energies).mean()
std_norm= norm_fit[1] #np.log(-mean_energies).std()

tot = 0
for i in range(8):
    f6sigma_g = mean_norm + i*std_norm
    f6sigma_l = mean_norm - i*std_norm
    greater_6sigma = np.where(np.log(-mean_energies)>f6sigma_g)
    less_6sigma = np.where(np.log(-mean_energies)<f6sigma_l)


    f7sigma_g = mean_norm + (i+1)*std_norm
    f7sigma_l = mean_norm - (i+1)*std_norm

    greater_7sigma = np.where(np.log(-mean_energies)>f7sigma_g)
    less_7sigma = np.where(np.log(-mean_energies)<f7sigma_l)


    # Find the indices of elements in array1 that are also in array2
    indices_to_remove_gt = np.isin(greater_6sigma[0], greater_7sigma[0])
    indices_to_remove_lt = np.isin(less_6sigma[0], less_7sigma[0])
    # print(indices_to_remove_gt)

    # Remove elements based on indices
    array3 = greater_6sigma[0][~indices_to_remove_gt]
    array4 = less_6sigma[0][~indices_to_remove_lt]

    all = np.concatenate((array3, array4))
    print(f'all between {f7sigma_l} and {f6sigma_l} and {f6sigma_g} and {f7sigma_g} = {len(all)}')

    tot+= len(all)
    print(len(all))
    for idx in all:
        energy_score_df.loc[idx, 'Interval mean'] = i+1
        #print(f"Index: {idx}, Mean: { energy_score_df.loc[idx, 'HMC RGZ']}, Std: {energy_score_df.loc[idx, 'HMC RGZ std']}")
    
print(tot)
energy_score_df.to_csv('radiogalaxies_bnns/results/ood/hmc_energy_scores_rgz_sigma.csv')

exit()

## Do the same for std of energies

std_energies = energy_score_df['HMC RGZ std']
lognorm_fit = stats.lognorm.fit(std_energies)

# Generate a range of values for plotting the fitted distribution
x = np.linspace(std_energies.min(), std_energies.max(), len(std_energies))
pdf = stats.lognorm.pdf(x, *lognorm_fit)

# Plot the histogram of the data
plt.clf()
plt.figure(dpi=300)
sns.histplot(std_energies, kde=False, stat='density', binwidth=0.1, label='Std energy')

# Plot the fitted lognormal distribution
plt.plot(x, pdf, 'r-', label='Fitted Lognormal Distribution')
plt.xlabel('Energy')
plt.ylabel('Density')
plt.legend()
plt.savefig('radiogalaxies_bnns/results/ood/energy_hist_hmc_rgz_fit_std.png')
