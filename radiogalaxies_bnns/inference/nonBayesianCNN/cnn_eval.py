import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb


import radiogalaxies_bnns.inference.utils as utils
import radiogalaxies_bnns.inference.nonBayesianCNN.utils as cnn_utils

from radiogalaxies_bnns.inference.models import LeNet
from radiogalaxies_bnns.inference.datamodules import MNISTDataModule, MiraBestDataModule, testloader_mb_uncert
from radiogalaxies_bnns.eval.uncertainty.uncertainty import entropy_MI, overlapping, GMM_logits, calibration



def credible_interval(samples, credibility):
    '''
    calculate credible interval - equi-tailed interval instead of highest density interval

    samples values and indices
    '''
    mean_samples = samples.mean()
    sorted_samples = np.sort(samples)
    lower_bound = 0.5 * (1 - credibility)
    upper_bound = 0.5 * (1 + credibility)

    index_lower = int(np.round(len(samples) * lower_bound))
    index_upper = int(np.round(len(samples) * upper_bound))

    # print('lower', index_lower, 'upper', index_upper)

    return sorted_samples, index_lower, index_upper, mean_samples

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_dict, config = utils.parse_config('/share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/inference/nonBayesianCNN/config_cnn.txt')
seed = config_dict['training']['seed']
torch.manual_seed(seed)
path_out = config_dict['output']['path_out']
criterion = torch.nn.CrossEntropyLoss()


# datamodule = MiraBestDataModule(config_dict, hmc=False)
# train_loader = datamodule.train_dataloader()
# validation_loader = datamodule.val_dataloader()
# test_loader = datamodule.test_dataloader()
print('TEST DATASET')
print(config_dict['output']['test_data'])
test_loader, test_data1, data_type, test_data = testloader_mb_uncert(config_dict['output']['test_data'], config_dict['data']['datadir'])

for i, (x_test, y_test) in enumerate(test_loader):
    x_test, y_test = x_test.to(device), y_test.to(device)
    
model = LeNet(in_channels = 1, output_size =  2).to(device)

n_ensembles = 10

softmax_ensemble, logits_ensemble = cnn_utils.eval_ensemble(model, test_loader, device, n_ensembles, path_out, criterion)
softmax_ensemble = torch.reshape(softmax_ensemble, (len(y_test), n_ensembles, 2))
logits_ensemble = torch.reshape(logits_ensemble, (len(y_test), n_ensembles, 2))


# print(logits_ensemble[0:1, :, :2][0])
# print(logits_ensemble[0:1, :, :2][0].shape)
# print(criterion(logits_ensemble[0:0+1, :, :2][0], torch.tile(y_test[0], (n_ensembles, 1)).flatten()))



avg_error_mean = []

error_all = []
entropy_all = []
mi_all = []
aleat_all = []

fr1 = 0
fr2 = 0

loss_all = [] 

color = []
for k in range(len(y_test)):
    # print('galaxy', k)

    x = [0, 1]
    x = np.tile(x, (n_ensembles, 1))

    target = y_test[k].cpu().detach().numpy()

    if(target == 0):
        fr1+= 1
    elif(target==1):
        fr2+= 1
    
    # logits = pred_list[burn:][:, k:(k+1), :2].detach().numpy()
    # softmax_values = F.softmax(pred_list[burn:][:, k:(k+1), :2], dim =-1).detach().numpy()

    softmax_values = softmax_ensemble[k:k+1, :, :2].cpu().detach().numpy()
    loss = criterion(logits_ensemble[k:k+1, :, :2][0], torch.tile(y_test[k], (n_ensembles, 1)).flatten())

    sorted_softmax_values, lower_index, upper_index, mean_samples = credible_interval(softmax_values[:, :, 0].flatten(), 1) #0.64
    upper_index = upper_index - 1
    # print("90% credible interval for FRI class softmax", sorted_softmax_values[lower_index], sorted_softmax_values[upper_index])
    
    sorted_softmax_values_fr1 = sorted_softmax_values[lower_index:upper_index]
    sorted_softmax_values_fr2 = 1 - sorted_softmax_values[lower_index:upper_index]

    softmax_mean = np.vstack((mean_samples, 1-mean_samples)).T
    softmax_credible = np.vstack((sorted_softmax_values_fr1, sorted_softmax_values_fr2)).T

    entropy, mutual_info, entropy_singlepass = entropy_MI(softmax_credible, 
                                                          samples_iter= len(softmax_credible[:,0]))



    pred_mean = np.argmax(softmax_mean, axis = 1)
    error_mean = (pred_mean != target)*1


    pred = np.argmax(softmax_credible, axis = 1)

    y_test_all = np.tile(target, len(softmax_credible[:,0]))

    errors =  np.mean((pred != y_test_all).astype('uint8'))

    avg_error_mean.append(error_mean)
    error_all.append(errors)
    entropy_all.append(entropy/np.log(2))    
    mi_all.append(mutual_info/np.log(2))
    aleat_all.append(entropy_singlepass/np.log(2))
    loss_all.append(loss.item())


print("mean and std of error")
print(error_all)
print(np.round(np.mean(error_all)*100, 3))
print(np.round(np.std(error_all), 3 ))


print("Average CE Loss")
print(loss_all)
print(np.round(np.mean(loss_all), 3))
print(np.round(np.std(loss_all), 3 ))

fr1_start = 0 #{0}
fr1_end = fr1 #{49, 68}
fr2_start = fr1 #49, 68
fr2_end = len(y_test) #len(val_indices) #{104, 145}
    
n_bins = 8
print('BINS', 8)
path = './'
uce  = calibration(path, np.array(error_all), np.array(entropy_all), n_bins, x_label = 'predictive entropy')
print("Predictive Entropy")
print("uce = ", np.round(uce, 2))


uce  = calibration(path, np.array(error_all), np.array(mi_all), n_bins, x_label = 'mutual information')
print("Mutual Information")
print("uce = ", np.round(uce, 2))


uce  = calibration(path, np.array(error_all), np.array(aleat_all), n_bins, x_label = 'average entropy')
print("Average Entropy")
print("uce = ", np.round(uce, 2))
