import torch
import hamiltorch

import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import InterpolationMode
from pytorch_lightning.demos.mnist_datamodule import MNIST
from torch.utils.data import DataLoader, random_split
from datamodules import MNISTDataModule, MiraBestDataModule, testloader_mb_uncert
from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl
import utils
from PIL import Image

from models import MLP, LeNet

from matplotlib import pyplot as plt
import mirabest
from uncertainty import entropy_MI, overlapping, GMM_logits, calibration
import csv
import seaborn as sns

def credible_interval(samples, credibility):
    '''
    calculate credible interval - equi-tailed interval instead of highest density interval

    samples values and indices
    '''
    sorted_samples = np.sort(samples)
    lower_bound = 0.5 * (1 - credibility)
    upper_bound = 0.5 * (1 + credibility)

    index_lower = int(np.round(len(samples) * lower_bound))
    index_upper = int(np.round(len(samples) * upper_bound))

    return sorted_samples, index_lower, index_upper
# hamiltorch.set_random_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LeNet(1, 2) #model = LeNet(1, 2, jobid=2)  #MLP(150, 200, 10)

path = './results/priors/thin1000/' #'./results/temp/thin1000/'  #'./results/checkpt/' #'./results/galahad/hamilt/testing/15000steps/'

params_hmc = torch.load(path+'/thin_chain'+str(2), map_location=torch.device(device)) #torch.load(path+'params_hmc_thin1000.pt', map_location = torch.device(device))

#loading multiple thinned chains together instea
# n_chains = 4
# for j in range(n_chains):
#     var_name = "params_hmc_{}".format(i)
#     locals()[var_name] = torch.load(path+'/thin_chain'+str(0), map_location=torch.device(device))

burn_in = 20
params_hmc = params_hmc[burn_in:]

# tau = 1 => std =1
# tau = 10 => std =0.31
# tau = 100 => std =0.1
# tau = 200 => std =0.07
# tau = 500 => std =0.04
# tau = 1000 => std =0.003

tau_list = []
tau = 100. # 10.#./100. # 1/50



for w in model.parameters():
#     print(w.nelement())
#     tau_list.append(tau/w.nelement())
    tau_list.append(tau)
tau_list = torch.tensor(tau_list).to(device)

config_dict, config = utils.parse_config('config_mb.txt')

datamodule = MiraBestDataModule(config_dict, hmc=True)
# train_loader = datamodule.train_dataloader()
# validation_loader = datamodule.val_dataloader()
# test_loader = datamodule.test_dataloader()

test_loader, test_data1, data_type, test_data = testloader_mb_uncert(config_dict['output']['test_data'], config_dict['data']['datadir'])

for i, (x_test, y_test) in enumerate(test_loader):
    x_test, y_test = x_test.to(device), y_test.to(device)

print(len(y_test))

#get predictions
pred_list, log_prob_list = hamiltorch.predict_model(model, x = x_test, y = y_test, samples=params_hmc, model_loss='multi_class_linear_output', tau_out=1., tau_list=tau_list)
_, pred = torch.max(pred_list, 2)

print(pred_list.shape) #([180, 104, 2])


# print('pred_list', pred_list[:, 0:1, :2])
# print('pred_list', pred_list[:, 102:103, :2].shape) #[180,1,2]
# softmax_list = F.softmax(pred_list[:, 102:103, :2], dim = -1)
# print('softmax values', softmax_list)
# print('softmax values', len(softmax_list[0]))

# print('preds' , pred)
# print('logprob list', log_prob_list)

# print('softmax values', F.softmax(pred_list, dim = -1))

_, softmax_preds = torch.max(F.softmax(pred_list, dim = -1), 2)
# print('softmax preds', softmax_preds)


y_test_tile = y_test.repeat(len(params_hmc), 1) 
# print(y_test_tile)

# print(softmax_preds == y_test_tile)

preds_int = (softmax_preds == y_test_tile).int() #int()).float32()
preds_int = preds_int.type(torch.float32)
# print(preds_int)


sum_per_galaxy  = torch.mean(preds_int, dim = 0 )
sum_per_sample = torch.mean(preds_int, dim = 1 )
# print(sum_per_galaxy) #sum over column
# print(sum_per_sample) #sum over row

print(torch.mean(sum_per_galaxy), torch.std(sum_per_galaxy))

print(torch.mean(sum_per_sample), torch.std(sum_per_sample))

# exit()

_, pred = torch.max(pred_list, 2)
acc = []
acc = torch.zeros( int(len(params_hmc))-1)
nll = torch.zeros( int(len(params_hmc))-1)
ensemble_proba = F.softmax(pred_list[0], dim=-1)

acc_mean = torch.zeros( int(len(params_hmc))-1)
acc_std = torch.zeros( int(len(params_hmc))-1)
#calculate accuracy and NLL

for s in range(1,len(params_hmc)):

    _, pred = torch.max(pred_list[:s].mean(0), -1)
    # print('pred_list', pred_list[:s])
    # print('pred', pred)

    acc[s-1] = (pred.float() == y_test.flatten()).sum().float()/y_test.shape[0]
    # std_iter[s-1] = (pred.float() == y_test.flatten()).sum().float()/y_test.shape[0]
    # print('acc', acc[s-1])
    ensemble_proba += F.softmax(pred_list[s], dim=-1) #ensemble prob is being added - save all the softmax values

    # print('softmax', F.softmax(pred_list[s], dim=-1))
    # print('max from softmax', torch.max(F.softmax(pred_list[s], dim=-1)))
    
    # print('accuracy', (torch.max(F.softmax(pred_list[s], dim=-1)) ==  y_test.flatten()))

    nll[s-1] = F.nll_loss(torch.log(ensemble_proba.cpu()/(s+1)), y_test[:].long().cpu().flatten(), reduction='mean')
    
    
    # acc_mean[s-1] = torch.mean(acc)
    
    # acc_std[s-1] = torch.std(acc)

    # print(acc[s-1], acc_mean[s-1], acc_std[s-1])

    # if(s%100==0):
    #     print('pred_list', pred_list[:s])
    #     print('pred_list[:s].shape',  pred_list[:s].shape)
    #     print('pred_list[:s].mean()', pred_list[:s].mean(0))
    #     print('pred_list[:s].std()', pred_list[:s].std(0))
    #     print('torch.max', torch.max(pred_list[:s].mean(0), -1))
    #     print('pred', pred)

    #     print("test accuracy", torch.mean(acc), torch.std(acc))
    #     print('ACCURACY', torch.mean(acc), '+/-',  torch.std(acc))
    #     print('NLL', torch.mean(nll), '+/-', torch.std(nll))
    #     # exit()

print('ACCURACY', torch.mean(acc), '+/-',  torch.std(acc))
print('NLL', torch.mean(nll), '+/-', torch.std(nll))

plt.figure(dpi=200)
plt.plot(acc, label = 'chain1')
plt.ylabel('accuracy')
plt.xlabel('step')
plt.legend()
plt.savefig(path+'acc_chains.png')

plt.figure(dpi=200)
plt.plot(nll, label = 'chain1')
plt.ylabel('negative log likelihood')
plt.xlabel('step')
plt.legend()
plt.savefig(path+'nll_chains.png')


# #calculate credible interval - equi-tailed interval instead of highest density interval
# sorted_acc = torch.sort(acc)[0]
# credible_interval = 0.90 #90% credible interval
# lower_bound = 0.5 * (1-credible_interval)
# upper_bound = 0.5 * (1+credible_interval)
# print(lower_bound, upper_bound)

# index_lower = int(np.round(len(params_hmc) * lower_bound))
# index_upper = int(np.round(len(params_hmc) * upper_bound))

# print(index_lower, index_upper)
# print(torch.sort(acc)[0][index_lower],torch.sort(acc)[0][index_upper])

# sorted_acc = torch.sort(acc)[0][index_lower:index_upper]

# sns.displot(torch.sort(acc)[0])
# plt.axvline(torch.sort(acc)[0][index_lower], color='orange', linestyle='--')
# plt.axvline(torch.sort(acc)[0][index_upper], color='orange', linestyle='--')
# plt.axvline(torch.sort(acc)[0][index_lower*2], color='green', linestyle='--') #not exactly
# plt.xlabel('Accuracy')
# plt.savefig(path+'dist of accuracy')


############################
# UCE Calculation
############################

error_all = []
entropy_all = []
mi_all = []
aleat_all =[]
    
fr1 = 0
fr2 = 0

indices = np.arange(0, len(test_data), 1)
print(indices)

#for each sample in the test set:
for index in indices:

    # x_test = x_test[index] #torch.unsqueeze(x_test[index])
    # y_test = y_test[index] #torch.unsqueeze(x_test[index])
    x_test = torch.unsqueeze(torch.tensor(test_data[index][0]),0)
    y_test = torch.unsqueeze(torch.tensor(test_data[index][1]),0)

    # x_test[0]
    # print(x_test, y_test)

    # target = y_test[index].detach().numpy()
    target = y_test.detach().numpy().flatten()[0]

    if(target == 0):
        fr1+= 1
    elif(target==1):
        fr2+= 1

    pred_list, log_prob_list = hamiltorch.predict_model(model, x = x_test, y = y_test, samples=params_hmc, model_loss='multi_class_linear_output', tau_out=1., tau_list=tau_list)
    softmax_ = F.softmax(pred_list, dim=-1)

    pred = softmax_.mean(dim=1).argmax(dim=-1).numpy().flatten()
    pred1 = pred_list.mean(dim=1).argmax(dim=-1)

    y_test_all = np.tile(y_test.detach().numpy().flatten()[0], len(params_hmc))

    # print(pred)
    # print(pred1)
    # print(y_test_all)

    # print(pred != y_test_all)

    errors =  np.mean((pred != y_test_all).astype('uint8'))
    # print(errors)

    softmax = np.array(softmax_).reshape((len(params_hmc), 2))
    logits = pred #np.array(pred) 

    mean_logits = np.mean(logits,axis=0)
    var_logits = np.std(logits,axis=0)
    entropy, mutual_info, entropy_singlepass = entropy_MI(softmax, samples_iter= len(params_hmc))
  

    error_all.append(errors)
    entropy_all.append(entropy/np.log(2))    
    mi_all.append(mutual_info/np.log(2))


# print(error_all)
# print(entropy_all)




fr1_start = 0 #{0}
fr1_end = fr1 #{49, 68}
fr2_start = fr1 #49, 68
fr2_end = len(indices) #len(val_indices) #{104, 145}
    
# print(fr1_start, fr1_end, fr2_start, fr2_end)
    
#print(error_all)
#entropy, mutual_info, entropy_singlepass = entropy_MI(softmax, samples_iter)
n_bins = 8
    
uce  = calibration(path, np.array(error_all), np.array(entropy_all), n_bins, x_label = 'predictive entropy')
print("Predictive Entropy")
print("uce = ", np.round(uce, 2))
uce_0  = calibration(path, np.array(error_all[fr1_start:fr1_end]), np.array(entropy_all[fr1_start:fr1_end]), n_bins, x_label = 'predictive entropy') 
print("UCE FRI= ", np.round(uce_0, 2))
uce_1  = calibration(path, np.array(error_all[fr2_start:fr2_end]), np.array(entropy_all[fr2_start:fr2_end]), n_bins, x_label = 'predictive entropy')
print("UCE FRII = ", np.round(uce_1, 2))
cUCE = (uce_0 + uce_1)/2 
print("cUCE=", np.round(cUCE, 2))
    
#max_mi = np.amax(np.array(mi_all))
#print("max mi",max_mi)
#mi_all = mi_all/max_mi
#print(mi_all)
#print("max mi",np.amax(mi_all))
uce  = calibration(path, np.array(error_all), np.array(mi_all), n_bins, x_label = 'mutual information')
print("Mutual Information")
print("uce = ", np.round(uce, 2))
uce_0  = calibration(path, np.array(error_all[fr1_start:fr1_end]), np.array(mi_all[fr1_start:fr1_end]), n_bins, x_label = 'mutual information') 
print("UCE FRI= ", np.round(uce_0, 2))
uce_1  = calibration(path, np.array(error_all[fr2_start:fr2_end]), np.array(mi_all[fr2_start:fr2_end]), n_bins, x_label = 'mutual information')
print("UCE FRII = ", np.round(uce_1, 2))
cUCE = (uce_0 + uce_1)/2 
print("cUCE=", np.round(cUCE,2))  

print("mean and std of error")
print(error_all)
print(np.round(np.mean(error_all)*100), 3)
print(np.round(np.std(error_all)), 3 )

# uce  = calibration(path, np.array(error_all), np.array(aleat_all), n_bins, x_label = 'average entropy')
# print("Average Entropy")
# print("uce = ", np.round(uce, 2))
# uce_0  = calibration(path, np.array(error_all[fr1_start:fr1_end]), np.array(aleat_all[fr1_start:fr1_end]), n_bins, x_label = 'average entropy') 
# print("UCE FRI= ",np.round(uce_0, 2))
# uce_1  = calibration(path, np.array(error_all[fr2_start:fr2_end]), np.array(aleat_all[fr2_start:fr2_end]), n_bins, x_label = 'average entropy')
# print("UCE FRII = ", np.round(uce_1, 2))
# cUCE = (uce_0 + uce_1)/2 
# print("cUCE=", np.round(cUCE, 2))  
    
exit()


#define output path to save softmax ditributions etc
csvfile = config_dict['output']['filename_uncert']   
path_out = config_dict['output']['path_out']
cvsfile = path_out + csvfile

rows = ['index', 'target', 'entropy', 'entropy_singlepass' , 'mutual info', 'var_logits_0','var_logits_1','softmax_eta', 'logits_eta','cov0_00','cov0_01','cov0_11','cov1_00','cov1_01','cov1_11', 'data type', 'label']
                                    
with open(csvfile, 'w+', newline="") as f_out:
    writer = csv.writer(f_out, delimiter=',')
    writer.writerow(rows)

#for each sample in the test set:
for j in range(len(y_test)):
    softmax_list = []
    logits_list = []
    for k in range(len(params_hmc)):
        logits_list.append(pred_list[k][j].detach().numpy().flatten())
        softmax_list.append(F.softmax(pred_list[k], dim=-1)[j].detach().numpy().flatten())

    x = [0, 1]
    x = np.tile(x, (len(params_hmc), 1))

    target = y_test[j].detach().numpy()
    if(target == 0):
        if(data_type == 'Hybrid'):
            label = 'Conf'
        else:
            label = 'FRI'
        
    elif(target ==1):
        if(data_type == 'Hybrid'):
            label = 'Uncert'
        else:
            label = 'FRII'
        

    plt.figure(figsize= (2.6, 4.8), dpi=300)
    plt.rcParams["axes.grid"] = False
    plt.subplot((211))
    plt.scatter(x, softmax_list, marker='_',linewidth=1,color='b',alpha=0.5)
    plt.title("softmax outputs")
    plt.xticks(np.arange(0, 2, 1.0), labels = ['I', 'II'])
        
    plt.subplot((212))
    plt.imshow(test_data1[j][0])
    plt.axis("off")
    plt.title(label)
    plt.savefig(path_out+ 'softmax_test' + str(j)+ '.png')

    softmax = np.array(softmax_list)
    logits = np.array(logits_list) 
  

    mean_logits = np.mean(logits,axis=0)
    var_logits = np.std(logits,axis=0)


    entropy, mutual_info, entropy_singlepass = entropy_MI(softmax, samples_iter= len(params_hmc))

    softmax_eta = overlapping(softmax[:,0], softmax[:,1])

    logits_eta = overlapping(logits[:,0], logits[:,1])
        
    covs = GMM_logits(logits, 2)

    plt.figure(dpi=300)
    covs = GMM_logits(logits, 2)
    plt.savefig(path_out+'cov' + str(j)+ '.png')
    
    plt.figure(dpi=200)
    plt.rcParams["axes.grid"] = False
    plt.axes().set_facecolor('white')
    plt.scatter(x, logits, marker='_',linewidth=1,color='b',alpha=0.5)
    plt.xticks(np.arange(0, 2, 1))
    plt.savefig(path_out+'logits' + str(j)+ '.png')

    # data_type = 'MBConf'
    # ['index', 'target', 'entropy','entropy_singlepass ', 'mutual info', 'var_logits_0','var_logits_1','softmax_eta', 'logits_eta','GMM_covs', 'data type', 'label']
    _results = [j, target, entropy, entropy_singlepass, mutual_info, var_logits[0],var_logits[1], softmax_eta, logits_eta,covs[0][0][0], covs[0][0][1], covs[0][1][1],covs[1][0][0],covs[1][0][1],covs[1][1][1], data_type, label ]
        
    with open(csvfile, 'a', newline="") as f_out:
        writer = csv.writer(f_out, delimiter=',')
        writer.writerow(_results)

exit()
