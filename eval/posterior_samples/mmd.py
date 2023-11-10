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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def z_statistic(x, y, num_params, num_samples):
    x_mean = torch.mean(x, dim = 0)
    y_mean = torch.mean(y, dim = 0)

    n_data_sqrt= np.sqrt(num_samples)

    x_std = torch.std(x, dim = 0)/n_data_sqrt
    y_std = torch.std(y, dim = 0)/n_data_sqrt

    Z = abs(x_mean - y_mean)/torch.sqrt(x_std*x_std + y_std*y_std)

    return Z

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

# # path = './results/inits/thin1000'    #leapfrog/' #inits/thin1000' # #priors/thin1000'
# n_chains = 1 #4 #4 #10
# # checkpt = 500

# #load n chains 
# for i in range(n_chains):
#     var_name = "params_hmc_{}".format(i)
#     locals()[var_name] = torch.load(path+'/thin_chain'+str(i), map_location=torch.device('cpu'))

# print( "len chain 1", len(params_hmc_0))


# #= [156, 2416, 10426, 20832, 188280, 10164, 170]
# num_params = 168 #5
# num_samples = len(params_hmc_0) #number of samples after thinning

# samples_corner0 =np.zeros((num_samples, num_params))

# i = 0
# # [156, 2572, 12998, 33830, 222110, 232274, 232444]
# # [ 150 , 156, 2556, 2572, 12972, 12998, 33798, 33830, 221990, 222110, 232190, 232274, 232442, 232444]

# for j in np.arange(232274, 232442):#232274, 232442
#     for k in range(num_samples):
#         samples_corner0[k][i] = params_hmc_0[k][j]

#     # print(j)
#     i=i+1

# samples_corner0 = 0

# np.append(samples_corner0, samples_corner1)


print(samples_hmc.shape)


# vi_name = 'vi_samples_lap_l7_weights.npy' #'vi_samples_lap_l1_weights.npy' #
# 
# path_vi = './results/vi/'
# vi_samples_laplace = np.load(path_vi+vi_name)
# vi_samples_laplace = vi_samples_laplace.reshape((num_samples, num_params_vi))#[:, 158:] #[:, 163:]

num_params_vi = 168
indices_vi = list(range(0, 168))
vi_samples_laplace = get_vi_samples(path_vi_laplace, num_samples, num_params_vi, indices_vi)
print(vi_samples_laplace.shape)

vi_samples_gaussian = get_vi_samples(path_vi_gaussian, num_samples, num_params_vi, indices_vi)
vi_samples_gmm = get_vi_samples(path_vi_gmm, num_samples, num_params_vi, indices_vi)


# vi_samples_gaussian = np.load(path_vi+'vi_samples_gaussian_l7_weights.npy') #'vi_samples_gaussian.npy'
# vi_samples_gaussian = vi_samples_gaussian.reshape((num_samples, num_params_vi))#[:, 158:] #[:, 163:]

# vi_samples_gmm = np.load(path_vi+'vi_samples_gmm_l7_weights.npy') #'vi_samples_gmm.npy'
# vi_samples_gmm = vi_samples_gmm.reshape((num_samples, num_params_vi))#[:, 158:]#[:, 163:]



# lla_samples =  torch.load('./results/laplace/lla_samples.pt', map_location=torch.device('cpu')).detach().numpy()
lla_samples = get_lla_samples(path_lla, indices_vi)

print(lla_samples.shape)

# dataframes
# vi_sample_laplace_df = pd.DataFrame(vi_samples_laplace, index = ['VI (Laplace prior)']*num_samples) 
# # vi_sample_laplace_df = pd.DataFrame(vi_samples_laplace[:num_samples : , ], index = ['VI (Laplace prior)']*num_samples) 
# vi_sample_gaussian_df = pd.DataFrame(vi_samples_gaussian, index = ['VI (Gaussian prior)']*num_samples) 
# vi_sample_gmm_df = pd.DataFrame(vi_samples_gmm, index = ['VI (GMM prior)']*num_samples) 
# lla_samples_df = pd.DataFrame(lla_samples[:, 163:], index = ['LLA samples']*num_samples) 



# z_stat = torch.zeros((num_params))

s1 = torch.from_numpy(samples_hmc)
s2 = torch.from_numpy(vi_samples_laplace)
s3 = torch.from_numpy(lla_samples)
s3 = s3.type(torch.float64)

s4 = torch.from_numpy(vi_samples_gaussian)
s5 = torch.from_numpy(vi_samples_gmm)

print(torch.mean(s1, dim = 0).shape)




z_stat_hmc_lap = z_statistic(s1, s2, num_params, num_samples)
print("Z_stat HMC and VI (laplace)" , z_stat_hmc_lap.shape)


z_stat_hmc_lla = z_statistic(s1, s3, num_params, num_samples)
# print("Z_stat HMC and LLA" ,z_stat_hmc_lla)


z_stat_hmc_gaussian = z_statistic(s1, s4, num_params, num_samples)
# print("Z_stat HMC and VI (gaussian)" ,z_stat_hmc_gaussian)


z_stat_hmc_gmm = z_statistic(s1, s5, num_params, num_samples)
# print("Z_stat HMC and VI (gmm)" ,z_stat_hmc_gmm)

z_stat_hmc_lap_df = pd.DataFrame({"HMC vs VI (Laplace)": z_stat_hmc_lap} )  #pd.DataFrame(z_stat_hmc_lap, columns={0: "HMC vs VI (Laplace)"}) 
z_stat_hmc_lla_df = pd.DataFrame({"HMC vs LLA": z_stat_hmc_lla }) #(z_stat_hmc_lla, columns={0: "HMC vs LLA"}) 
z_stat_hmc_gaussian_df = pd.DataFrame({"HMC vs VI (Gaussian)": z_stat_hmc_gaussian })  #(z_stat_hmc_gaussian, columns={0: "HMC vs VI (Gaussian)"}) 
z_stat_hmc_gmm_df = pd.DataFrame({"HMC vs VI (GMM)": z_stat_hmc_gmm })   #(z_stat_hmc_gmm, columns={0: "HMC vs VI (GMM)"}) 

z_stat_combined_df = pd.concat([z_stat_hmc_lla_df, z_stat_hmc_lap_df, 
                                z_stat_hmc_gaussian_df, 
                                z_stat_hmc_gmm_df
                                ], axis=1).reset_index()

# .rename(columns={0: "HMC vs LLA", 
#                                   1: "HMC vs VI (Laplace)",
#                                   2: "HMC vs VI (Gaussian)",
#                                   3: "HMC vs VI (GMM)",                             
#                                   })
# 
# z_stat_combined_df = pd.DataFrame([z_stat_hmc_lla, z_stat_hmc_lap, z_stat_hmc_gaussian, z_stat_hmc_gmm])

print(z_stat_combined_df)
x = np.arange(0, 168, 1)
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


# z_stat = z_statistic(s1, s1, num_params, num_samples)
# print("Z_stat HMC and HMC" ,z_stat)

exit()
for i in range(num_params):
    s1 = torch.from_numpy(samples_corner0[:, i])
    s2 = torch.from_numpy(vi_samples_laplace[:, i])

    s1_mean = torch.mean(s1)
    s2_mean = torch.mean(s2)

    n_data_sqrt= np.sqrt(200)

    s1_std = torch.std(s1)/n_data_sqrt
    s2_std = torch.std(s2)/n_data_sqrt


    Z = (s1_mean - s2_mean)/torch.sqrt(s1_std*s1_std + s2_std*s2_std)
    z_stat[i] = Z

exit()
# plt.hist(z_stat)
# plt.savefig('./z_stat.png')


# from sklearn.neighbors import KernelDensity
# i =0
# kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(samples_corner0[:, i])
# log_density = kde.score_samples(samples_corner0[:, i])
# print(log_density)

# i = 0
# print (kl_div(samples_corner0[:, i], vi_samples_laplace[:, i]))
# print (kl_div(samples_corner0[:, i], lla_samples[:, i]) )


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

# from torch.distributions.multivariate_normal import MultivariateNormal

# m = 20 # sample size
# x_mean = torch.zeros(2)+1
# y_mean = torch.zeros(2)
# x_cov = 2*torch.eye(2) # IMPORTANT: Covariance matrices must be positive definite
# y_cov = 3*torch.eye(2) - 1

# px = MultivariateNormal(x_mean, x_cov)
# qy = MultivariateNormal(y_mean, y_cov)
# x = px.sample([m]).to(device)
# y = qy.sample([m]).to(device)

# print(x)

# i= 0

# s1 = torch.from_numpy(samples_corner0[:, i])
# s2 = torch.from_numpy(vi_samples_laplace[:, i])
# s3 = torch.from_numpy(lla_samples[:, i])

s1 = torch.from_numpy(samples_corner0)
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
