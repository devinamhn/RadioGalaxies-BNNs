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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = './results/inits/thin1000'    #leapfrog/' #inits/thin1000' # #priors/thin1000'
n_chains = 1 #4 #4 #10
# checkpt = 500


'''
Make corner plots
First five of last layer: (232358,232363) ; num_params = 5
Last five of last layer: (232437,232442) ; num_params = 5   (232435,232440)
bias last layer:(232442, 232444) ; num_params = 2
all 84 of the last layer: (232358, 232442) ; num_params = 84
first 5 conv layer weights: (0,5)? 
'''
#load n chains 
for i in range(n_chains):
    var_name = "params_hmc_{}".format(i)
    locals()[var_name] = torch.load(path+'/thin_chain'+str(i), map_location=torch.device('cpu'))

print( "len chain 1", len(params_hmc_0))
# print( "len chain 2", len(params_hmc_1))
# print( "len chain 3", len(params_hmc_2))
# print( "len chain 5", len(params_hmc_3))

#= [156, 2416, 10426, 20832, 188280, 10164, 170]
num_params = 5 #168 #5
num_samples = len(params_hmc_0) #number of samples after thinning

samples_corner0 =np.zeros((num_samples, num_params))
samples_corner1 = np.zeros((num_samples, num_params))

samples_corner2 = np.zeros((num_samples, num_params))
samples_corner3 = np.zeros((num_samples, num_params))

# samples_corner4 = np.zeros((num_samples, num_params))
# samples_corner5 = np.zeros((num_samples, num_params))
# samples_corner6 = np.zeros((num_samples, num_params))
# samples_corner7 = np.zeros((num_samples, num_params))
# samples_corner8 = np.zeros((num_samples, num_params))
# samples_corner9 = np.zeros((num_samples, num_params))

i = 0
# [156, 2572, 12998, 33830, 222110, 232274, 232444]
# [ 150 , 156, 2556, 2572, 12972, 12998, 33798, 33830, 221990, 222110, 232190, 232274, 232442, 232444]

# for j in np.arange( 232274 , 232442): [232274 + 6, 232274 + 15, 232274 + 35, 232274 + 39, 232274 + 41 ]
for j in [232274 + 6, 232274 + 15, 232274 + 35, 232274 + 39, 232274 + 41 ]: #np.arange(232274, 232279): #  (232274, 232279)  (232437 , 232442)
# for j in np.arange(232440, 232442): #232435,232440 (232437,232442): 
#for j in np.arange (16736, 16741): 
# for j in np.arange (0, 5):
    for k in range(num_samples):

        # var_name = "samples_corner{}".format(k)
        # locals()[var_name] = 

        samples_corner0[k][i] = params_hmc_0[k][j]
        # samples_corner1[k][i] = params_hmc_1[k][j]
        # samples_corner2[k][i] = params_hmc_2[k][j]
        # samples_corner3[k][i] = params_hmc_3[k][j]

    print(j)
    i=i+1

# np.append(samples_corner0, samples_corner1)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

# samples_corner0 = softmax(samples_corner0)
# samples_corner1 = softmax(samples_corner1)

print(samples_corner0.shape)
# samples_corner0 = samples_corner0.reshape((num_params, num_samples))
# print(np.cov(samples_corner0).shape)
# samples_0_cov = np.cov(samples_corner0)


# plt.figure(dpi = 200)
# sns.heatmap(samples_0_cov)
# plt.title('Layer7')
# plt.savefig('./heatmap_cov_l7_weights.png')


index_highz = [6, 15, 35, 39, 41]

vi_name = 'vi_samples_lap_l7_weights.npy' #'vi_samples_last5.npy'
num_params_vi = 168
path_vi = './results/vi/'
vi_samples_laplace = np.load(path_vi+vi_name)
vi_samples_laplace = vi_samples_laplace.reshape((num_samples, num_params_vi)) #[:, 0:5] #[:, 163:]
vi_samples_laplace = np.take(vi_samples_laplace, indices = index_highz, axis=1)# #torch.index_select(vi_samples_laplace, dim = 1, index = index_highz)
print(vi_samples_laplace.shape)


# vi_samples_gaussian = np.load(path_vi+'vi_samples_gaussian_l7_weights.npy') #'vi_samples_gaussian.npy'
# vi_samples_gaussian = vi_samples_gaussian.reshape((num_samples, num_params_vi))[:, 163:] #[:, 163:]

# vi_samples_gmm = np.load(path_vi+'vi_samples_gmm_l7_weights.npy') #'vi_samples_gmm.npy'
# vi_samples_gmm = vi_samples_gmm.reshape((num_samples, num_params_vi))[:, 163:]#[:, 163:]

vi_sample_laplace_df = pd.DataFrame(vi_samples_laplace, index = ['VI (Laplace prior)']*num_samples) 
# vi_sample_laplace_df = pd.DataFrame(vi_samples_laplace[:num_samples : , ], index = ['VI (Laplace prior)']*num_samples) 
# vi_sample_gaussian_df = pd.DataFrame(vi_samples_gaussian, index = ['VI (Gaussian prior)']*num_samples) 
# vi_sample_gmm_df = pd.DataFrame(vi_samples_gmm, index = ['VI (GMM prior)']*num_samples) 


##trying to make better plots with seaborn 
## figure out how to plot seaborn plts of top of each other 
# column = num_params, row = num_samples
burnin = 0
samples0_df = pd.DataFrame(samples_corner0[burnin:], index = ['HMC']*(num_samples-burnin))
# samples0_df_burn = pd.DataFrame(samples_corner0[:burnin], index = ['HMC burnin']*burnin)
# samples1_df = pd.DataFrame(samples_corner1)
# samples2_df = pd.DataFrame(samples_corner2)
# samples3_df = pd.DataFrame(samples_corner3)

lla_samples =  torch.load('./results/laplace/lla_samples.pt', map_location=torch.device('cpu')).detach().numpy()
lla_samples = np.take(lla_samples, indices = index_highz, axis=1)
print(lla_samples.shape)

# lla_samples_df = pd.DataFrame(lla_samples[:, 0:5], index = ['LLA samples']*num_samples) 
lla_samples_df = pd.DataFrame(lla_samples, index = ['LLA samples']*num_samples) 



map_weights = np.load('./dropout/map_samples_l7_weights.npy')
# print(np.take(map_weights, index_highz))
map_weight_highz= np.take(map_weights, index_highz)
map_samples=  np.tile(map_weight_highz, (200, 1))
print(map_samples.shape)

dropout_weights = np.load('./dropout/dropout_samples_l7_weights.npy')
# print(np.take(map_weights, index_highz))
dropout_weight_highz= np.take(dropout_weights, index_highz)
dropout_samples=  np.tile(dropout_weight_highz, (200, 1))
print(dropout_samples.shape)


map_samples_df = pd.DataFrame(map_samples, index=['MAP value']*num_samples)
dropout_samples_df = pd.DataFrame(dropout_samples, index=['Dropout']*num_samples)
frames = [samples0_df, vi_sample_laplace_df, lla_samples_df, map_samples_df, dropout_samples_df] #, vi_sample_gaussian_df, vi_sample_gmm_df] #[samples0_df, samples0_df, vi_sample_laplace_df] #, vi_sample_gaussian_df, vi_sample_gmm_df]
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

plt.savefig('./results/snspairplt_l7_highz_kde_all.png')

exit()

figure = corner.corner(
    samples_corner0,
    labels=[ 
        # r"$w_{1 \_ 1}^1$",
        # r"$w_{1 \_ 2}^1$",
        # r"$w_{1 \_ 3}^1$",
        # r"$w_{1 \_ 4}^1$",
        # r"$w_{1 \_ 5}^1$", 

        # r"$w_{2 \_ 1}^7$",
        # r"$w_{2 \_ 2}^7$",
        # r"$w_{2 \_ 3}^7$",
        # r"$w_{2 \_ 4}^7$",
        # r"$w_{2 \_ 5}^7$", 

        r"$w_{2 \_ 80}^7$",
        r"$w_{2 \_ 81}^7$",
        r"$w_{2 \_ 82}^7$",
        r"$w_{2 \_ 83}^7$",
        r"$w_{2 \_ 84}^7$",

        # r"$w_{1}^8$",
        # r"$w_{2}^8$",
    ],
    # quantiles=[0.16, 0.5, 0.84],
    # titles=['MCMC'],
    show_titles=True,
    title_kwargs={"fontsize": 12},
    # truths=samples_corner0.mean(0),
    # truth_color='black',
    color = 'black',
    plot_datapoints = True,
    plot_contours = False

)

# corner.corner(samples_corner[0:9000:20][:], quantiles=[0.16, 0.5, 0.84],fig = figure, color='b', show_titles = True)

# corner.corner(samples_corner1, 
#             #   quantiles=[0.16, 0.5, 0.84],
#               fig = figure, color='y', 
#               show_titles = True,
#               plot_datapoints = True,
#               plt_contours = False)

# corner.corner(samples_corner2, 
#               quantiles=[0.16, 0.5, 0.84],
#               fig = figure, color='g', 
#               show_titles = True)

# corner.corner(samples_corner3, 
#             #   quantiles=[0.16, 0.5, 0.84],
#               fig = figure, 
#               color='r', 
#               show_titles = True)

# corner.corner(samples_corner4, quantiles=[0.16, 0.5, 0.84],fig = figure, color='b', show_titles = True)


# save without VI results
# plt.savefig(path+'/corner_mb_chains_last2_beforesoftmax.png')

# plt.savefig(path+'/corner_mb_chains_lastlayer.png')

# exit()
'''
VI samples
'vi_samples_first5.npy'  'corner_overlaid_first5.png'
'vi_samples_last5.npy'  'corner_overlaid_last5.png'
'vi_samples_bias.npy' 'corner_overlaid_lastlayer_bias.png'
'''
# vi_name = 'vi_samples_first5.npy'
# corner_name =  'corner_overlaid_first5.png'

vi_name = 'vi_samples_last5.npy'
# corner_name =  'corner_overlaid_first5_compare.png'

path_vi = './results/vi/'
vi_samples = np.load(path_vi+vi_name)

corner.corner(vi_samples, fig=figure,  
            #   quantiles=[0.16, 0.5, 0.84], 
              color ='m',
              #titles=['Laplace'],
              show_titles=True
    )


vi_samples_gaussian = np.load(path_vi+'vi_samples_gaussian.npy')

corner.corner(vi_samples_gaussian, 
              fig=figure,  
            #   quantiles=[0.16, 0.5, 0.84], 
              color ='c',
    #titles=['Gaussian'],
    show_titles=True,
    )

vi_samples_gmm = np.load(path_vi+'vi_samples_gmm.npy')

corner.corner(vi_samples_gmm, 
              fig=figure,  
            #   quantiles=[0.16, 0.5, 0.84], 
              color ='orange',
    #titles=['GMM'],
    show_titles=True,
    )
plt.savefig('./corner_pts.png')
# plt.savefig(path+'/corner_mb_random_200samples_last.png')

exit()

label1 = 'chain 1' #'init std = 0.01'
label2 = 'chain 2' #'init std = 0.1'
# label3 = 'chain 3'
# label4 = 'chain 4'
# print(np.array(samples_corner0[0][:]))
# print(np.array(samples_corner0[0][:]).shape)
# print(np.array(params_hmc_0).shape)

num_sample = len(params_hmc_0) #25000
sample0=np.zeros(num_sample)
sample1=np.zeros(num_sample)
# sample2=np.zeros(num_sample)
# sample3=np.zeros(num_sample)
# sample4=np.zeros(num_sample)
# sample5=np.zeros(num_sample)
# sample6=np.zeros(num_sample)
# sample7=np.zeros(num_sample)
# sample8=np.zeros(num_sample)
# sample9=np.zeros(num_sample)

# print(params_hmc_0.shape)

for i in range(num_sample):
    sample0[i] = params_hmc_0[i][230000] #16741
    sample1[i] = params_hmc_1[i][230000]
    # sample2[i] = params_hmc_2[i][230000] #16741
    # sample3[i] = params_hmc_3[i][230000]
    # sample4[i] = params_hmc_4[i][232437]
    # sample5[i] = params_hmc_5[i][232437]
    # sample6[i] = params_hmc_6[i][232437]
    # sample7[i] = params_hmc_7[i][232437]
    # sample8[i] = params_hmc_8[i][232437]
    # sample9[i] = params_hmc_9[i][232437]


# print(sample0[0])
# print(sample1[0])
# print(sample2[0])
# print(sample3[0])


plt.figure(figsize = ((10,5)) , dpi=200)
plt.plot(np.array(sample0), label = label1) #L - number of steps per trajectory
plt.plot(np.array(sample1), label = label2) 
# plt.plot(np.array(sample2), label = label3) 
# plt.plot(np.array(sample3), label = label4) 
# plt.plot(np.array(sample4), label = label4) 
# plt.plot(np.array(sample5), label = label4) 
# plt.plot(np.array(sample6), label = label4) 
# plt.plot(np.array(sample7), label = label4) 
# plt.plot(np.array(sample8), label = label4) 
# plt.plot(np.array(sample9), label = label4) 

# plt.plot(np.array(sample1), label = label2) 
plt.legend()
plt.savefig(path+'/trace_chains.png')

plt.figure(dpi=200)
plot_acf(sample0, lags = None, label = label1)
plt.title('Autocorrelation')
plt.legend()
plt.savefig(path+'/acorr_1.png')

plt.figure(dpi=200)
plot_acf(sample1, lags = None, label = label2)
plt.title('Autocorrelation')
plt.legend()
plt.savefig(path+'/acorr_2.png')

# plt.figure(dpi=200)
# plot_acf(sample2, lags = None, label = label3)
# plt.title('Autocorrelation')
# plt.legend()
# plt.savefig(path+'/acorr_3.png')

# plt.figure(dpi=200)
# plot_acf(sample3, lags = None, label = label4)
# plt.title('Autocorrelation')
# plt.legend()
# plt.savefig(path+'/acorr_5.png')



