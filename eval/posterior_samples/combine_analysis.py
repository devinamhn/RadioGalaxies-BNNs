import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_hmc_samples(path, n_chains, param_indices, chain_index):
    '''
    Get samples from a single HMC chain

    param_indices : list of ints or ndarray generated from np.arange

    indices_per_layer = [156, 2572, 12998, 33830, 222110, 232274, 232444] \\weights and biases combined
    indices_per_layer = [150 , 156, 2556, 2572, 12972, 12998, 33798, 33830, 221990, 222110, 232190, 232274, 232442, 232444] \\ weights, biases per layer
    high z-statistic indices = [232274 + 6, 232274 + 15, 232274 + 35, 232274 + 39, 232274 + 41 ]
    np.arange(232437 , 232442)
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
    
def plot_heatmap(samples, filename):
    '''
    plot heat map to visualise covariance matrix of weights
    '''
    samples_cov = np.cov(samples)

    plt.figure(dpi = 200)
    sns.heatmap(samples_cov)
    plt.title('Covariance Map')
    plt.savefig(filename)


path_hmc = '/share/nas2/dmohan/mcmc/hamilt/results/inits/thin1000'    #leapfrog/' #inits/thin1000' # #priors/thin1000'
n_chains = 1 
chain_index = 0
param_indices = [232274 + 6, 232274 + 15, 232274 + 35, 232274 + 39, 232274 + 41 ]
num_params = len(param_indices)

samples_hmc, num_samples = get_hmc_samples(path_hmc, n_chains, param_indices, chain_index)
print(samples_hmc.shape)


# plot_heatmap(samples_hmc.reshape((num_params, num_samples)), filename = './heatmap_cov_l7_weights.png')

index_highz = [6, 15, 35, 39, 41]
num_params_vi = 168
path_vi_laplace = '/share/nas2/dmohan/mcmc/hamilt/results/vi/' + 'vi_samples_lap_l7_weights.npy' 
path_vi_gaussian = '/share/nas2/dmohan/mcmc/hamilt/results/vi/' + 'vi_samples_gaussian_l7_weights.npy' 
path_vi_gmm = '/share/nas2/dmohan/mcmc/hamilt/results/vi/' + 'vi_samples_gmm_l7_weights.npy' 



vi_samples_laplace = get_vi_samples(path_vi_laplace, num_samples, num_params_vi, index_highz)
vi_samples_gaussian = get_vi_samples(path_vi_gaussian, num_samples, num_params_vi, index_highz)
vi_samples_gmm = get_vi_samples(path_vi_gmm, num_samples, num_params_vi, index_highz)

print(vi_samples_laplace.shape)


#create dataframes: column = num_params, row = num_samples

burnin = 0
hmc_samples_df = pd.DataFrame(samples_hmc[burnin:], index = ['HMC']*(num_samples-burnin))
# samples0_df_burn = pd.DataFrame(samples_corner0[:burnin], index = ['HMC burnin']*burnin)  #for plotting burnin samples separately

vi_sample_laplace_df = pd.DataFrame(vi_samples_laplace, index = ['VI (Laplace prior)']*num_samples) 
# vi_sample_laplace_df = pd.DataFrame(vi_samples_laplace[:num_samples : , ], index = ['VI (Laplace prior)']*num_samples) 
# vi_sample_gaussian_df = pd.DataFrame(vi_samples_gaussian, index = ['VI (Gaussian prior)']*num_samples) 
# vi_sample_gmm_df = pd.DataFrame(vi_samples_gmm, index = ['VI (GMM prior)']*num_samples) 

def get_lla_samples(path, indices):
    lla_samples =  torch.load(path, map_location=torch.device('cpu')).detach().numpy()
    lla_samples = np.take(lla_samples, indices, axis=1)
    return lla_samples

# lla_samples =  torch.load('/share/nas2/dmohan/mcmc/hamilt/results/laplace/lla_samples.pt', map_location=torch.device('cpu')).detach().numpy()
# lla_samples = np.take(lla_samples, indices = index_highz, axis=1)

path_lla = '/share/nas2/dmohan/mcmc/hamilt/results/laplace/lla_samples.pt'
lla_samples = get_lla_samples(path_lla, index_highz)
print(lla_samples.shape)

# lla_samples_df = pd.DataFrame(lla_samples[:, 0:5], index = ['LLA samples']*num_samples) 
lla_samples_df = pd.DataFrame(lla_samples, index = ['LLA samples']*num_samples) 

def get_dropout_samples(path, indices, num_samples):

    '''
    Function to get samples from dropout and MAP training 

    '''
    weights = np.load(path)
    weights_= np.take(map_weights, indices)
    weight_samples =  np.tile(weights, (num_samples, 1))

    return weight_samples

path_map = '/share/nas2/dmohan/mcmc/hamilt/dropout/map_samples_l7_weights.npy'
path_dropout = '/share/nas2/dmohan/mcmc/hamilt/dropout/dropout_samples_l7_weights.npy'

map_samples = get_dropout_samples(path_map, index_highz, num_samples)
print(map_samples.shape)

dropout_samples = get_dropout_samples(path_dropout, index_highz, num_samples)
print(dropout_samples.shape)

# map_weights = np.load('/share/nas2/dmohan/mcmc/hamilt/dropout/map_samples_l7_weights.npy')
# map_weight_highz= np.take(map_weights, index_highz)
# map_samples=  np.tile(map_weight_highz, (200, 1))


# dropout_weights = np.load('/share/nas2/dmohan/mcmc/hamilt/dropout/dropout_samples_l7_weights.npy')
# dropout_weight_highz= np.take(dropout_weights, index_highz)
# dropout_samples=  np.tile(dropout_weight_highz, (200, 1))

exit()

map_samples_df = pd.DataFrame(map_samples, index=['MAP value']*num_samples)
dropout_samples_df = pd.DataFrame(dropout_samples, index=['Dropout']*num_samples)

frames = [hmc_samples_df, vi_sample_laplace_df, lla_samples_df, map_samples_df, dropout_samples_df] #, vi_sample_gaussian_df, vi_sample_gmm_df] #[samples0_df, samples0_df, vi_sample_laplace_df] #, vi_sample_gaussian_df, vi_sample_gmm_df]
newdf = pd.concat(frames).rename(columns=
                                 
                                 {0: r"$w_{1 \_ 6}^7$", 
                                  1: r"$w_{1 \_ 15}^7$",
                                  2: r"$w_{1 \_ 35}^7$",
                                  3: r"$w_{1 \_ 39}^7$",
                                  4: r"$w_{1 \_ 41}^7$"                                  
                                  }
                                #  {0: r"$w_{1 \_ 1}^7$", 
                                #   1: r"$w_{1 \_ 2}^7$",
                                #   2: r"$w_{1 \_ 3}^7$",
                                #   3: r"$w_{1 \_ 4}^7$",
                                #   4: r"$w_{1 \_ 5}^7$"                                  
                                #   }


                                #  {0: r"$w_{2 \_ 80}^7$", 
                                #   1: r"$w_{2 \_ 81}^7$",
                                #   2: r"$w_{2 \_ 82}^7$",
                                #   3: r"$w_{2 \_ 83}^7$",
                                #   4: r"$w_{2 \_ 84}^7$"                                  
                                #   }
                                  
                                  ).reset_index()

print(newdf)

# sns.pairplot(newdf,  corner=True, diag_kind='kde', hue = 'index')
# plt.savefig('./results/snskde_l7.png')


# sns.pairplot(samples0_df,  corner=True, diag_kind='kde')
# # sns.pairplot(samples1_df,  corner=True)
# g = sns.pairplot(samples2_df, corner=True)
# g.pairplot(samples3_df,  corner=True)

# print(newdf)


g = sns.PairGrid(newdf, hue = 'index', diag_sharey=False)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot)
# plt.axvline()

i = 0
for ax in g.axes.ravel():
    if(i == 0):
        ax.axvline(x=map_weight_highz[0], color = "red")
        ax.axvline(x=dropout_weight_highz[0], color = "violet")
    if(i == 6):
        ax.axvline(x=map_weight_highz[1], color = "red")
        ax.axvline(x=dropout_weight_highz[1], color = "violet")
    if(i == 12):
        ax.axvline(x=map_weight_highz[2], color = "red")
        ax.axvline(x=dropout_weight_highz[2], color = "violet")
    if(i == 18):
        ax.axvline(x=map_weight_highz[3], color = "red")
        ax.axvline(x=dropout_weight_highz[3], color = "violet")
    if(i == 24):
        ax.axvline(x=map_weight_highz[4], color = "red")
        ax.axvline(x=dropout_weight_highz[4], color = "violet")
    i+=1

plt.savefig('./snspairplt_l7_highz_kde_all.png')

exit()




