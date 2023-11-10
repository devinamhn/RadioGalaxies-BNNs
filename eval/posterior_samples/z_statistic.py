import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from utils import get_hmc_samples, get_vi_samples, get_lla_samples, get_dropout_samples

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


def z_statistic(x, y, num_params, num_samples):
    x_mean = torch.mean(x, dim = 0)
    y_mean = torch.mean(y, dim = 0)

    n_data_sqrt= np.sqrt(num_samples)

    x_std = torch.std(x, dim = 0)/n_data_sqrt
    y_std = torch.std(y, dim = 0)/n_data_sqrt

    Z = abs(x_mean - y_mean)/torch.sqrt(x_std*x_std + y_std*y_std)

    return Z

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


path_hmc = '/share/nas2/dmohan/mcmc/hamilt/results/inits/thin1000'    #leapfrog/' #inits/thin1000' # #priors/thin1000'
path_vi_laplace = '/share/nas2/dmohan/mcmc/hamilt/results/vi/' + 'vi_samples_lap_l7_weights.npy' 
path_vi_gaussian = '/share/nas2/dmohan/mcmc/hamilt/results/vi/' + 'vi_samples_gaussian_l7_weights.npy' 
path_vi_gmm = '/share/nas2/dmohan/mcmc/hamilt/results/vi/' + 'vi_samples_gmm_l7_weights.npy' 
path_lla = '/share/nas2/dmohan/mcmc/hamilt/results/laplace/lla_samples.pt'
path_map = '/share/nas2/dmohan/mcmc/hamilt/dropout/map_samples_l7_weights.npy'
path_dropout = '/share/nas2/dmohan/mcmc/hamilt/dropout/dropout_samples_l7_weights.npy'


n_chains = 1 
chain_index = 0
param_indices = np.arange(232274, 232442)
num_params = len(param_indices)

samples_hmc, num_samples = get_hmc_samples(path_hmc, n_chains, param_indices, chain_index)

num_params_vi = 168
indices_vi = list(range(0, 168))
vi_samples_laplace = get_vi_samples(path_vi_laplace, num_samples, num_params_vi, indices_vi)
vi_samples_gaussian = get_vi_samples(path_vi_gaussian, num_samples, num_params_vi, indices_vi)
vi_samples_gmm = get_vi_samples(path_vi_gmm, num_samples, num_params_vi, indices_vi)
lla_samples = get_lla_samples(path_lla, indices_vi)


print(samples_hmc.shape)
print(vi_samples_laplace.shape)
print(lla_samples.shape)



s1 = torch.from_numpy(samples_hmc)
s2 = torch.from_numpy(vi_samples_laplace)
s3 = torch.from_numpy(lla_samples) #lla samples need to be converted to float 64
s3 = s3.type(torch.float64)
s4 = torch.from_numpy(vi_samples_gaussian)
s5 = torch.from_numpy(vi_samples_gmm)



z_stat_hmc_lap = z_statistic(s1, s2, num_params, num_samples)
z_stat_hmc_lla = z_statistic(s1, s3, num_params, num_samples)
z_stat_hmc_gaussian = z_statistic(s1, s4, num_params, num_samples)
z_stat_hmc_gmm = z_statistic(s1, s5, num_params, num_samples)

z_stat_hmc_lap_df = pd.DataFrame({"HMC vs VI (Laplace)": z_stat_hmc_lap} ) 
z_stat_hmc_lla_df = pd.DataFrame({"HMC vs LLA": z_stat_hmc_lla })
z_stat_hmc_gaussian_df = pd.DataFrame({"HMC vs VI (Gaussian)": z_stat_hmc_gaussian })
z_stat_hmc_gmm_df = pd.DataFrame({"HMC vs VI (GMM)": z_stat_hmc_gmm })

z_stat_combined_df = pd.concat([z_stat_hmc_lla_df, z_stat_hmc_lap_df, 
                                z_stat_hmc_gaussian_df, 
                                z_stat_hmc_gmm_df
                                ], axis=1).reset_index()


print(z_stat_combined_df)

x = np.arange(0, num_params, 1)
plt.figure(dpi=200, figsize=(30,4))

sns.scatterplot(z_stat_combined_df, x = 'index', y  = 'HMC vs LLA')
sns.scatterplot(z_stat_combined_df, x = 'index', y  = 'HMC vs VI (Laplace)')
# sns.scatterplot(z_stat_combined_df, x = 'index', y  = 'HMC vs VI (Gaussian)')
# sns.scatterplot(z_stat_combined_df, x = 'index', y  = 'HMC vs VI (GMM)')
plt.xticks(np.arange(0, 168, 2 ))
plt.ylabel('Z statistic')
plt.title("Z-statistic for the last layer weights")
plt.axhline(y=2, color = 'black', alpha = 0.2)
plt.axhline(y=3, color = 'black', alpha = 0.2)
plt.legend(['HMC vs LLA','HMC vs VI (Laplace)' ])
plt.savefig('./z_stat_hmc_lap.png')

exit()