import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from utils import get_hmc_samples, get_vi_samples, get_lla_samples, get_dropout_samples

def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    # xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    print(x.dtype, y.t().dtype)
    zz = torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)



    return torch.mean(XX + YY - 2. * XY)

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
s3 = torch.from_numpy(lla_samples)
s3 = s3.type(torch.float64)

s4 = torch.from_numpy(vi_samples_gaussian)
s5 = torch.from_numpy(vi_samples_gmm)

# print(s1, s2, s3.dtype)
# s1_mean = torch.mean(s1)
# s2_mean = torch.mean(s2)
# s3_mean = torch.mean(s3)

mmd = MMD(s1, s2, kernel='multiscale')
print("MMD HMC and VI (laplace)", mmd)

mmd = MMD(s1, s4, kernel='multiscale')
print("MMD HMC and VI (gaussian)", mmd)

mmd = MMD(s1, s5, kernel='multiscale')
print("MMD HMC and VI (gmm)", mmd)

mmd = MMD(s1, s3, kernel='multiscale')
print("MMD HMC and LLA", mmd)
