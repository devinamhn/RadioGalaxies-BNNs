
import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def get_hmc_samples(path, n_chains, param_indices, chain_index):
    '''
    Get samples from a single HMC chain

    param_indices : list of ints or ndarray generated from np.arange

    indices_per_layer = [156, 2572, 12998, 33830, 222110, 232274, 232444] \\weights and biases combined
    indices_per_layer = [150 , 156, 2556, 2572, 12972, 12998, 33798, 33830, 221990, 222110, 232190, 232274, 232442, 232444] \\ weights, biases per layer
    '''
    num_params = len(param_indices)

    params_hmc = torch.load(path+'/thin_chain'+str(chain_index), map_location=torch.device('cpu'))

    num_samples = len(params_hmc)
    samples_hmc = np.zeros((num_samples, num_params))
    
    i = 0
    for j in param_indices:
        for k in range(num_samples):
            samples_hmc[k][i] = params_hmc[k][j]
        print(j)
        i=i+1    

    return samples_hmc, num_samples


def get_vi_samples(path_vi, num_samples, num_params_vi, indices):

    vi_samples = np.load(path_vi)
    vi_samples = vi_samples.reshape((num_samples, num_params_vi)) #[:, 0:5] #[:, 163:]
    vi_samples = np.take(vi_samples, indices, axis=1)# #torch.index_select(vi_samples, dim = 1, index = index_highz)

    return vi_samples

def get_lla_samples(path, indices):
    lla_samples =  torch.load(path, map_location=torch.device('cpu')).detach().numpy()
    lla_samples_indexed = np.take(lla_samples, indices, axis=1)
    return lla_samples_indexed


    
def plot_heatmap(samples, filename):
    '''
    plot heat map to visualise covariance matrix of weights
    '''
    samples_cov = np.cov(samples)

    plt.figure(dpi = 200)
    sns.heatmap(samples_cov)
    plt.title('Covariance Map')
    plt.savefig(filename)

def get_dropout_samples(path, indices, num_samples):

    '''
    Function to get samples from dropout and MAP training 
    Change dropout samples later

    '''
    weights = np.load(path)
    weights_indexed= np.take(weights, indices)
    weight_samples =  np.tile(weights_indexed, (num_samples, 1))

    return weights_indexed, weight_samples
