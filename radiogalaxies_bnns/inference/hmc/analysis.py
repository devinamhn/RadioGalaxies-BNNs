import torch
# import numpyro
from matplotlib import pyplot as plt
import corner
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
import pandas as pd

path ='./results/mnist/mnist_mlp_no_mass_9000samples/'
path2= path + 'diff_random_seed/' #'initstd1/'  #'./results/checkpt/' #'./results/galahad/hamilt/testing/15000steps/'
num_samples = 9000
params_hmc = torch.load(path + 'params_hmc.pt', map_location=torch.device('cpu'))
params_hmc1 = torch.load(path2 + 'params_hmc.pt',map_location=torch.device('cpu'))

# params_hmc = params_hmc0 + params_hmc1
# print(len(params_hmc))

#print(params_hmc[6999][:].shape)

# ####################################################################################
# ''' 
# Plot nll and accuracy
# '''

# acc_1 = torch.load(path + 'acc_train.pt')
# acc_2 = torch.load(path2 + 'acc_train.pt')

# nll_1 = torch.load(path + 'nll_train.pt')
# nll_2 = torch.load(path2 + 'nll_train.pt')

# logprob_1 = torch.load(path + 'log_prob_list.pt', map_location=torch.device('cpu'))
# logprob_2 = torch.load(path2 + 'log_prob_list.pt', map_location=torch.device('cpu'))

# burn_in = 0
# # print(torch.mean(acc_1[burn_in:]))
# # print(torch.mean(acc_2[burn_in:]))
# # print(np.mean([torch.mean(acc_1), torch.mean(acc_2)]))
# # print(np.std([torch.mean(acc_1), torch.mean(acc_2)]))

# # print(torch.mean(nll_1[burn_in:]))
# # print(torch.mean(nll_2[burn_in:]))
# # print(np.mean([torch.mean(nll_1), torch.mean(nll_2)]))
# # print(np.std([torch.mean(nll_1), torch.mean(nll_2)]))

label1 = 'rs1' #'init std = 0.01'
label2 = 'rs2' #'init std = 0.1'

# path = path + 'diff_random_seed/'
# plt.figure(dpi=200)
# plt.plot(acc_1, label = label1)
# plt.plot(acc_2, label = label2)
# plt.ylabel('accuracy')
# plt.xlabel('step')
# plt.legend()
# plt.savefig(path+'acc_chains.png')

# plt.figure(dpi=200)
# plt.plot(nll_1, label = label1)
# plt.plot(nll_2, label = label2)
# plt.ylabel('negative log likelihood')
# plt.xlabel('step')
# plt.legend()
# plt.savefig(path+'nll_chains.png')

# plt.figure(dpi=200)
# plt.plot(logprob_1, label = label1)
# plt.plot(logprob_2, label = label2)
# plt.ylabel('log probability')
# plt.xlabel('step')
# plt.legend()
# plt.savefig(path+'log_prob_chains.png')

# exit()
'''
Look at  the trace plot, histogram and autocorrelation plot of one of the weight params

# '''
num_sample = 9000 #25000
sample=np.zeros(num_sample)
sample1=np.zeros(num_sample)

# # print(params_hmc[99][232443])

for i in range(num_sample):
    sample[i] = params_hmc[i][16000] #16741
    sample1[i] = params_hmc1[i][16000]

path = path + 'diff_random_seed/'


plt.figure(dpi=200)
plt.plot(np.array(sample), label = label1) #L - number of steps per trajectory
plt.plot(np.array(sample1), label = label2) 
plt.savefig(path+'trace_chains.png')

exit()
# plt.figure(dpi=200)
# plt.hist(sample)
# plt.savefig(path+'hist.png')
# print(sample[2000:].size)

# plt.figure(dpi=200)
# plot_acf(sample[7000:], lags = 1000, label = 'burnin = 7000')
# plt.title('Lag = 1000; burnin = 7000')
# plt.savefig(path+'/initstd1/acorr_1000lag_7000burnin.png')

# plt.figure(dpi=200)
# plot_acf(sample, lags = 50, label = 'burnin = 1000')
# plt.title('Lag = 50; burnin = 1000')
# plt.savefig(path+'acorr_50lag_1000burnin.png')

# plt.figure(dpi=200)
# plot_acf(sample, lags = 100, label = 'burnin = 1000')
# plt.title('Lag = 100; burnin = 1000')
# plt.savefig(path+'acorr_100lag_1000burnin.png')

# plt.figure(dpi=200)
# plot_acf(sample[1000:], label = 'burnin = 2000')
# plt.title('Lag = acorr; burnin = 2000')
# plt.savefig(path+'acorr_20lag_2000.png')

# plt.figure(dpi=200)
# plot_acf(sample[2000:], label = 'burnin = 3000')
# plt.title('Lag = acorr; burnin = 3000')
# plt.savefig(path+'acorr_20lag_3000.png')


# plt.figure(dpi=200)
# plot_acf(sample[3000:], label = 'burnin = 4000')
# plt.title('Lag = acorr; burnin = 4000')
# plt.savefig(path+'acorr_20lag_burns_4000.png')

# ###################################################################################

'''
Make corner plots
First five of last layer: (232358,232363) ; num_params = 5
Last five of last layer: (232437,232442) ; num_params = 5
bias last layer:(232442, 232444) ; num_params = 2
all 84 of the last layer: (232358, 232442) ; num_params = 84
first 5 conv layer weights: (0,5)? 
'''

num_params = 5
samples_corner =np.zeros((num_samples, num_params))
samples_corner1 = np.zeros((num_samples, num_params))
i = 0

#for j in np.arange (232437,232442):
for j in np.arange (16736, 16741): 
    for k in range(num_samples):
        samples_corner[k][i] = params_hmc[k][j]
        samples_corner1[k][i] = params_hmc1[k][j]
    print(j)
    i=i+1
# print(samples_corner.shape)
# print(samples_corner.mean(0))

# print(samples_corner[0:9000:20][:].shape)
# exit()
figure = corner.corner(
    samples_corner,
    labels=[ 
    # #     r"$w_{1 \_ 1}^1$",
    # #     r"$w_{1 \_ 2}^1$",
    # #     r"$w_{1 \_ 3}^1$",
    # #     r"$w_{1 \_ 4}^1$",
    # #     r"$w_{1 \_ 5}^1$", 

    #     # r"$w_{2 \_ 1}^7$",
    #     # r"$w_{2 \_ 2}^7$",
    #     # r"$w_{2 \_ 3}^7$",
    #     # r"$w_{2 \_ 4}^7$",
    #     # r"$w_{2 \_ 5}^7$", 

    #     r"$w_{2 \_ 80}^7$",
    #     r"$w_{2 \_ 81}^7$",
    #     r"$w_{2 \_ 82}^7$",
    #     r"$w_{2 \_ 83}^7$",
    #     r"$w_{2 \_ 84}^7$",

    r"$w_{2 \_ 196}^3$",
    r"$w_{2 \_ 197}^3$",
    r"$w_{2 \_ 198}^3$",
    r"$w_{2 \_ 199}^3$",
    r"$w_{2 \_ 200}^3$",

    #     # r"$b_{1}^7$",
    #     # r"$b_{2}^7$",
    ],
    quantiles=[0.16, 0.5, 0.84],
    # titles=['MCMC'],
    show_titles=True,
    title_kwargs={"fontsize": 12},
    truths=samples_corner.mean(0),
    truth_color='black',
    color = 'black'
)

# corner.corner(samples_corner[0:9000:20][:], quantiles=[0.16, 0.5, 0.84],fig = figure, color='b', show_titles = True)

corner.corner(samples_corner1, quantiles=[0.16, 0.5, 0.84],fig = figure, color='y', show_titles = True)

plt.savefig(path+'corner_mnist_inits.png')

exit()
'''
'vi_samples_first5.npy'  'corner_overlaid_first5.png'
'vi_samples_last5.npy'  'corner_overlaid_last5.png'
'vi_samples_bias.npy' 'corner_overlaid_lastlayer_bias.png'
'''
# vi_name = 'vi_samples_first5.npy'
# corner_name =  'corner_overlaid_first5.png'

vi_name = 'vi_samples_last5.npy'
corner_name =  'corner_overlaid_last5_compare.png'

# vi_name ='vi_samples_bias.npy'
# corner_name =  'corner_overlaid_lastlayer_bias.png'
path = './results/vi/'
vi_samples = np.load(path+vi_name)

corner.corner(vi_samples, fig=figure,  quantiles=[0.16, 0.5, 0.84], color ='b',
    #titles=['Laplace'],
    show_titles=True
    )


vi_samples_gaussian = np.load(path+'vi_samples_gaussian.npy')

corner.corner(vi_samples_gaussian, fig=figure,  quantiles=[0.16, 0.5, 0.84], color ='g',
    #titles=['Gaussian'],
    show_titles=True,
    )

vi_samples_gmm = np.load(path+'vi_samples_gmm.npy')

corner.corner(vi_samples_gmm, fig=figure,  quantiles=[0.16, 0.5, 0.84], color ='y',
    #titles=['GMM'],
    show_titles=True,
    )
path = './results/checkpt/'
plt.savefig(path+'all.png')
#plt.savefig(path+corner_name)
exit()
####################################################################################


# ensemble = torch.load(path+'ensemble_prob.pt',map_location=torch.device('cpu'))
# print(ensemble.shape)

# print(torch.zeros( (int(200)-1, 104)).shape) #len(params_hmc)



# L = int(50)
# #my_list[start_index:end_index:step_size]
# # print(np.array(params_hmc[1::50]).shape)#((232444, 100)))
# plt.figure(dpi=200)
# plt.plot(params_hmc[0][1::L][:400]) #L - number of steps per trajectory
# plt.savefig(path+'temp.png')



#print(numpyro.effective_sample_size(samples[0][:]))

i = 0
num_params = 500
samples_conv =np.zeros((num_samples, num_params))#,dtype = 'uint8')

for j in np.arange(0, num_params): 
    for k in range(num_samples):
        samples_conv[k][i] = params_hmc[k][j]
    i=i+1

mean_hmc= np.mean(samples_conv, axis =0)
std_hmc= np.std(samples_conv, axis =0)

vi_mu_gaussian_linear = np.load(path+'vi_means_gaussian_linear.npy')
vi_mu_gaussian_conv = np.load(path+'vi_means_gaussian_conv.npy')

vi_mu_gmm_linear = np.load(path+'vi_means_gmm_linear.npy')
vi_mu_gmm_conv = np.load(path+'vi_means_gmm_conv.npy')

vi_mu_lap_linear = np.load(path+'vi_means_lap_linear.npy')
vi_mu_lap_conv = np.load(path+'vi_means_lap_conv.npy')

scatter_gaussian= (vi_mu_gaussian_linear[:num_params] - mean_hmc)/std_hmc
scatter_gmm= (vi_mu_gmm_linear[:num_params] - mean_hmc)/std_hmc
scatter_lap= (vi_mu_lap_linear[:num_params] - mean_hmc)/std_hmc
# print(scatter)
print(scatter_gaussian)
print("Gaussian", np.mean(scatter_gaussian), np.std(scatter_gaussian))
print("GMM", np.mean(scatter_gmm), np.std(scatter_gmm))
print("Laplace", np.mean(scatter_lap), np.std(scatter_lap))

exit()
weights_linear = np.arange(0, 33830)

fig = plt.figure(figsize=(20, 6))

plt.scatter(weights_linear[:num_params], scatter_gaussian, s=1, color = 'blue', label='Gaussian')
plt.scatter(weights_linear[:num_params], scatter_gmm, s=1, color = 'red', label='GMM')
plt.scatter(weights_linear[:num_params], scatter_lap, s=1, color = 'green', label='Laplace')
plt.legend()
# plt.ylim(-1, 1)
plt.axhline(y= 0, linestyle='dashed', color = 'black')
plt.hist(scatter_gaussian, orientation='horizontal', color='blue')
plt.hist(scatter_gmm, orientation='horizontal', color='red')
plt.hist(scatter_lap, orientation='horizontal', color='green')
plt.savefig(path+'temp_norm.png')

#linear: 198614
# data = pd.DataFrame({'x':weights_linear[:num_params], 'gaussian': scatter_gaussian, 'gmm': scatter_gmm,
#                     'lap': scatter_lap})
# sns.jointplot(data, x='x', y='gaussian', color='blue')
# sns.jointplot(data, x='x', y='gmm', color= 'red')
# sns.jointplot(data, x='x', y='lap', color= 'green')

plt.savefig(path+'temp_norm.png')

exit()

def scatter_hist(x, y, ax, ax_histy, s, color, label):
    # no labels
    # ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, s=s, color=color)#, label=label)

    # now determine nice limits by hand:
    binwidth = 0.2
    #xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    ymax = np.max(np.abs(y))
    lim = (int(ymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth) #np.arange(-lim, lim + binwidth, binwidth)
    #ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal', color=color)

# Start with a square Figure.
fig = plt.figure(figsize=(20, 6))
# Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
# Create the Axes.
ax = fig.add_subplot(gs[1, 0])
#ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
# Draw the scatter plot and marginals.
s= 1
color= 'blue'
label= 'Gaussian'
scatter_hist(weights_linear[:num_params], scatter_gaussian, ax, ax_histy, s, color, label)
color= 'red'
label= 'GMM'
scatter_hist(weights_linear[:num_params], scatter_gmm, ax, ax_histy, s, color, label)
color= 'green'
label= 'Laplace'
scatter_hist(weights_linear[:num_params], scatter_lap, ax, ax_histy, s, color, label)
plt.axhline(y= 0, linestyle='dashed', color = 'black')
plt.savefig(path+'temp_norm.png')

exit()
