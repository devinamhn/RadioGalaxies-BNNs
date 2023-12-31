import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from radiogalaxies_bnns.eval.posterior_samples.utils import get_hmc_samples, get_vi_samples, get_lla_samples, get_dropout_samples
from radiogalaxies_bnns.eval.posterior_samples.kld_estimators import scipy_estimator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


path_hmc = '/share/nas2/dmohan/mcmc/hamilt/results/inits/thin1000'    #leapfrog/' #inits/thin1000' # #priors/thin1000'
path_vi_laplace = '/share/nas2/dmohan/mcmc/hamilt/results/vi/' + 'vi_samples_lap_l7_weights.npy' 
path_vi_gaussian = '/share/nas2/dmohan/mcmc/hamilt/results/vi/' + 'vi_samples_gaussian_l7_weights.npy' 
path_vi_gmm = '/share/nas2/dmohan/mcmc/hamilt/results/vi/' + 'vi_samples_gmm_l7_weights.npy' 
path_lla = '/share/nas2/dmohan/mcmc/hamilt/results/laplace/lla_samples.pt'
path_map = '/share/nas2/dmohan/mcmc/hamilt/dropout/map_samples_l7_weights.npy'
path_dropout = '/share/nas2/dmohan/mcmc/hamilt/dropout/dropout_samples_l7_weights.npy'


n_chains = 1 
chain_index = 0
param_indices = [232274 + 1, 232274 + 14, 232274 + 36, 232274 + 39, 232274 + 41 ] #[232274 + 6, 232274 + 15, 232274 + 35, 232274 + 39, 232274 + 41 ]
num_params = len(param_indices)
# [1, 14, 36, 46, 68]
samples_hmc, num_samples = get_hmc_samples(path_hmc, n_chains, param_indices, chain_index)

index_highz = [1, 14, 36, 39, 41]  #[6, 15, 35, 39, 41]
num_params_vi = 168

vi_samples_laplace = get_vi_samples(path_vi_laplace, num_samples, num_params_vi, index_highz)
vi_samples_gaussian = get_vi_samples(path_vi_gaussian, num_samples, num_params_vi, index_highz)
vi_samples_gmm = get_vi_samples(path_vi_gmm, num_samples, num_params_vi, index_highz)
lla_samples = get_lla_samples(path_lla, index_highz)
map_weights, map_samples = get_dropout_samples(path_map, index_highz, num_samples)
dropout_weights, dropout_samples = get_dropout_samples(path_dropout, index_highz, num_samples)


print(samples_hmc.shape)
print(vi_samples_laplace.shape)
print(lla_samples.shape)
print(map_samples.shape)
print(dropout_samples.shape)

cov_vi_laplace = torch.cov(torch.from_numpy(vi_samples_laplace).reshape((5, 200)))
cov_vi_gaussian = torch.cov(torch.from_numpy(vi_samples_gaussian).reshape((5, 200)))
cov_hmc = torch.cov(torch.from_numpy(samples_hmc).reshape((5, 200)))

diagonal_shrinkage = cov_hmc/cov_vi_laplace


corr_hmc = torch.corrcoef(torch.from_numpy(samples_hmc).reshape((5, 200)))
corr_vi = torch.corrcoef(torch.from_numpy(vi_samples_laplace).reshape((5, 200)))


print('Correlation coeff hmc')
print(corr_hmc)
print('Correlation coeff vi')
print(corr_vi)


print('Diagonal shrinkage')
print(torch.diagonal(diagonal_shrinkage))

print('hmc diag cov', torch.diagonal(cov_hmc))
print('vi laplace diag cov', torch.diagonal(cov_vi_laplace))

print('hmc cov', cov_hmc)
print('vi laplace cov', cov_vi_laplace)

# print()
# print()

# exit()

#create dataframes: column = num_params, row = num_samples
burnin = 0
hmc_samples_df = pd.DataFrame(samples_hmc[burnin:], index = ['HMC']*(num_samples-burnin))
# samples0_df_burn = pd.DataFrame(samples_corner0[:burnin], index = ['HMC burnin']*burnin)  #for plotting burnin samples separately
vi_sample_laplace_df = pd.DataFrame(vi_samples_laplace, index = ['VI (Laplace prior)']*num_samples) 
# vi_sample_laplace_df = pd.DataFrame(vi_samples_laplace[:num_samples : , ], index = ['VI (Laplace prior)']*num_samples) 
vi_sample_gaussian_df = pd.DataFrame(vi_samples_gaussian, index = ['VI (Gaussian prior)']*num_samples) 
# vi_sample_gmm_df = pd.DataFrame(vi_samples_gmm, index = ['VI (GMM prior)']*num_samples) 
lla_samples_df = pd.DataFrame(lla_samples, index = ['LLA samples']*num_samples) #pd.DataFrame(lla_samples[:, 0:5], index = ['LLA samples']*num_samples) 
map_samples_df = pd.DataFrame(map_samples, index=['MAP value']*num_samples)
dropout_samples_df = pd.DataFrame(dropout_samples, index=['Dropout']*num_samples)

# samples_list = [hmc_samples_df, vi_sample_laplace_df, lla_samples_df, map_samples_df, dropout_samples_df]
samples_list = [hmc_samples_df, vi_sample_laplace_df]
bnn_samples_df = pd.concat(samples_list).rename(columns=
                                 
                                 {0: r"$w_{1 \_ 1}^7$", 
                                  1: r"$w_{1 \_ 14}^7$",
                                  2: r"$w_{1 \_ 36}^7$",
                                  3: r"$w_{1 \_ 39}^7$",
                                  4: r"$w_{1 \_ 41}^7$"                                  
                                  }
                                  # {0: r"$w_{1 \_ 6}^7$", 
                                  # 1: r"$w_{1 \_ 15}^7$",
                                  # 2: r"$w_{1 \_ 35}^7$",
                                  # 3: r"$w_{1 \_ 39}^7$",
                                  # 4: r"$w_{1 \_ 41}^7$"                                  
                                  # }
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

# print(bnn_samples_df)

#save dataframe for commonly used indices - last 5 of last layer, high z statistic values etc

g = sns.PairGrid(bnn_samples_df, hue = 'index', diag_sharey=False) #warn_singular=False
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot)
g.set(xlim=(-0.4,0.4), ylim = (-0.4,0.4))




i = 0
for ax in g.axes.ravel():
  ax.annotate("r_vi = {:.2f}".format(corr_vi.numpy().flatten()[i]), xy=(.65, .97), xycoords=ax.transAxes)
  ax.annotate("r_hmc = {:.2f}".format(corr_hmc.numpy().flatten()[i]), xy=(.65, .90), xycoords=ax.transAxes)
  # if(i == 0):
  #     ax.axvline(x=map_weights[0], color = "red")
  #     ax.axvline(x=dropout_weights[0], color = "violet")
  # if(i == 6):
  #     ax.axvline(x=map_weights[1], color = "red")
  #     ax.axvline(x=dropout_weights[1], color = "violet")
  # if(i == 12):
  #     ax.axvline(x=map_weights[2], color = "red")
  #     ax.axvline(x=dropout_weights[2], color = "violet")
  # if(i == 18):
  #     ax.axvline(x=map_weights[3], color = "red")
  #     ax.axvline(x=dropout_weights[3], color = "violet")
  # if(i == 24):
  #     ax.axvline(x=map_weights[4], color = "red")
  #     ax.axvline(x=dropout_weights[4], color = "violet")
  i+=1

plt.savefig('./snspairplt_vi_mixed.png')

# plot_heatmap(samples_hmc.reshape((num_params, num_samples)), filename = './heatmap_cov_l7_weights.png')

# print(samples_hmc[:, 0].reshape(num_samples, 1))

# for i in range(5):
#     kld = scipy_estimator(samples_hmc[:, i].reshape(num_samples, 1), vi_samples_laplace[:, i].reshape(num_samples, 1), 5)
#     print('kld hmc|vi', kld)

#     kld = scipy_estimator(samples_hmc[:, i].reshape(num_samples, 1), lla_samples[:, i].reshape(num_samples, 1), 5)
#     print('kld hmc|lla', kld)

exit()




