import torch
# torch.multiprocessing.set_sharing_strategy("file_system")
import os
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy
import numpy as np
import corner 
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import torch.nn.functional as F
import seaborn as sns
import pandas as pd
import arviz as az
import xarray as xr

import statsmodels.api as sm



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = './results/inits/thin1000' #inits/thin1000' # #priors/thin1000'
n_chains = 2 #4 #4 #10
# checkpt = 500

#load n chains 
for i in range(n_chains):
    var_name = "params_hmc_{}".format(i)
    locals()[var_name] = torch.load(path+'/thin_chain'+str(i), map_location=torch.device('cpu'))

print( "len chain 1", len(params_hmc_0))
print( "len chain 2", len(params_hmc_1))



num_params = 2 #5
num_samples = len(params_hmc_0) #number of samples after thinning

samples_corner0 =np.zeros((num_samples, num_params))
samples_corner1 = np.zeros((num_samples, num_params))

samples_corner2 = np.zeros((num_samples, num_params))
samples_corner3 = np.zeros((num_samples, num_params))

i = 0


'''
Make corner plots
First five of last layer: (232358,232363) ; num_params = 5
Last five of last layer: (232437,232442) ; num_params = 5   (232435,232440)
bias last layer:(232442, 232444) ; num_params = 2
all 84 of the last layer: (232358, 232442) ; num_params = 84
first 5 conv layer weights: (0,5)? 
'''

for j in np.arange(232440, 232442 ):
# for j in np.arange(232440, 232442): #232435,232440 (232437,232442): 
#for j in np.arange (16736, 16741): 
# for j in np.arange (0, 5):
    for k in range(num_samples):

        # var_name = "samples_corner{}".format(k)
        # locals()[var_name] = 

        samples_corner0[k][i] = params_hmc_0[k][j]
        samples_corner1[k][i] = params_hmc_1[k][j]
        # samples_corner2[k][i] = params_hmc_2[k][j]
        # samples_corner3[k][i] = params_hmc_3[k][j]

    print(j)
    i=i+1

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

vi_name = 'vi_samples_last5.npy'

path_vi = './results/vi/'
vi_samples_laplace = np.load(path_vi+vi_name)
vi_samples_gaussian = np.load(path_vi+'vi_samples_gaussian.npy')
vi_samples_gmm = np.load(path_vi+'vi_samples_gmm.npy')


vi_sample_laplace_df = pd.DataFrame(vi_samples_laplace[:num_samples : , ], index = ['VI (Laplace prior)']*num_samples) 
vi_sample_gaussian_df = pd.DataFrame(vi_samples_gaussian[:num_samples : , ], index = ['VI (Gaussian prior)']*num_samples) 
vi_sample_gmm_df = pd.DataFrame(vi_samples_gmm[:num_samples : , ], index = ['VI (GMM prior)']*num_samples) 

burnin = 0
samples0_df = pd.DataFrame(samples_corner0[burnin:])#, index = ['HMC']*(num_samples-burnin))
samples0_df_burn = pd.DataFrame(samples_corner0[:burnin], index = ['HMC burnin']*burnin)

frames = [samples0_df] # , samples0_df_burn] #, vi_sample_laplace_df, vi_sample_gaussian_df, vi_sample_gmm_df]
# newdf = pd.concat(frames).rename(columns={0: r"$w_{2 \_ 80}^7$", 
#                                   1: r"$w_{2 \_ 81}^7$",
#                                   2: r"$w_{2 \_ 82}^7$",
#                                   3: r"$w_{2 \_ 83}^7$",
#                                   4: r"$w_{2 \_ 84}^7$"                                  
#                                   }).reset_index()
newdf = pd.concat(frames)#.reset_index()

newdf["chain"] = 0
newdf["draw"] = np.arange(num_samples, dtype=int)
newdf = newdf.set_index(["chain", "draw"])
xdata = xr.Dataset.from_dataframe(newdf)
dataset = az.InferenceData(posterior=xdata)
dataset

# dataset = az.convert_to_inference_data(newdf)
# dataset = az.convert_to_inference_data(samples_corner0.reshape((1, )))

print(dataset)

# coords = {"school": ["Choate", "Deerfield"]}
az.plot_pair(
    dataset,
    # var_names=['~index', 0, 1, 2, 3, 4 ],
    kind="kde",
    # coords=coords,
    textsize=22,
    # kde_kwargs={
    #     "hdi_probs": [0.3, 0.6, 0.9],  # Plot 30%, 60% and 90% HDI contours
    #     "contourf_kwargs": {"cmap": "Blues"},
    # },
)
plt.savefig('./arviz_pairplot.png')




exit()