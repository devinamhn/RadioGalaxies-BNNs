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
from scipy.stats import shapiro, anderson

from scipy import stats
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = './results/inits/thin1000' #inits/thin1000' # #priors/thin1000'
n_chains = 1 #4 #4 #10
# checkpt = 500

#load n chains 
for i in range(n_chains):
    var_name = "params_hmc_{}".format(i)
    locals()[var_name] = torch.load(path+'/thin_chain'+str(i), map_location=torch.device('cpu'))

print( "len chain 1", len(params_hmc_0))




num_samples = len(params_hmc_0)
num_params = 232444
samples_corner0 =np.zeros((num_samples, num_params))



i = 0
for j in range(232444): #np.arange(200000, 200001): 
    for k in range(num_samples):
        samples_corner0[k][i] = params_hmc_0[k][j]
    # print(j)
    i=i+1

print(samples_corner0.shape)

# results = anderson(samples_corner0.flatten())
# print(results)
# print(results.statistic)
# print(results.critical_values)
# print(results.significance_level)



# ##################### fitting a Gaussian ######################################
# x_data = samples_corner0

# #plotting the histogram
# hist, bin_edges = np.histogram(x_data)
# hist=hist/sum(hist)
# # plt.savefig('./hist_fit_normal.png')

# n = len(hist)
# x_hist=np.zeros((n),dtype=float) 
# for ii in range(n):
#     x_hist[ii]=(bin_edges[ii+1]+bin_edges[ii])/2
    
# y_hist=hist

# #Calculating the Gaussian PDF values given Gaussian parameters and random variable X
# def gaus(X,C,X_mean,sigma):
#     return C*np.exp(-(X-X_mean)**2/(2*sigma**2))

# mean = sum(x_hist*y_hist)/sum(y_hist)                  
# sigma = sum(y_hist*(x_hist-mean)**2)/sum(y_hist) 

# #Gaussian least-square fitting process
# param_optimised,param_covariance_matrix = curve_fit(gaus,x_hist,y_hist,
#                                                     p0=[max(y_hist),mean,sigma],
#                                                     maxfev=5000)


# fig = plt.figure()
# x_hist_2=np.linspace(np.min(x_hist),np.max(x_hist),500)
# plt.plot(x_hist_2,gaus(x_hist_2,*param_optimised),'r.:',label='Gaussian fit')
# plt.legend()

# #Normalise the histogram values
# weights = np.ones_like(x_data) / len(x_data)
# plt.hist(x_data, weights=weights)

# #setting the label,title and grid of the plot
# plt.xlabel("Data: Random variable")
# plt.ylabel("Probability")
# plt.grid("on")
# plt.savefig('./hist_fit_normal.png')

################## shapiro tests

params_per_layer = [156, 2416, 10426, 20832, 188280, 10164, 170]
index_per_layer = [0, 156, 2572, 12998, 33830, 222110, 232274, 232444] #[0, 2, 4, 6, 8, 10, 12, 14, 16]
#

statistic_per_layer = np.zeros(7)
pvalue_per_layer = np.zeros(7)

for i in range(7):
    statistic_per_param = np.zeros(params_per_layer[i])
    pvalue_per_param = np.zeros(params_per_layer[i])
    critical_value_per_param = np.zeros(params_per_layer[i])

    k = 0
    for j in range(index_per_layer[i], index_per_layer[i+1]):
        results = shapiro(samples_corner0[:, j])
        # results =anderson(samples_corner0[:, j].flatten())

        statistic_per_param[k] = results.statistic
        # for shapiro test
        pvalue_per_param[k] = results.pvalue

        print(statistic_per_param[k])
        print(pvalue_per_param[k])

        # for anderson test
        # critical_value_per_param[k] = results.critical_values 
        k+=1

        # print(critical_value_per_param[k])
        


    statistic_per_layer[i] = statistic_per_param.mean()
    pvalue_per_layer[i] = pvalue_per_param.mean()

print(statistic_per_layer)
print(pvalue_per_layer)

# statistic_per_layer= np.array([0.99223917, 0.99251972, 0.99256914, 0.99256014, 0.99255771, 0.99258516,
#  0.99249917])
# pvalue_per_layer = np.array([0.46928904, 0.48832384, 0.49693193, 0.49493163, 0.49426775, 0.49711555,
#  0.49467731])


# plt.figure(figsize=(8, 4), dpi= 200)
# plt.scatter(np.arange(1, 8), statistic_per_layer, label = 'statistic')
# plt.scatter(np.arange(1, 8), pvalue_per_layer, label = 'pvalue')
# plt.legend(loc = 'upper left')
# plt.xlabel('Layer')
# plt.ylabel('Normality')
# plt.title('Shapiro test')
# plt.savefig('./normlaity.png')

################## shapiro tests end

exit()
