import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
import scipy
import pandas as pd
import seaborn as sns
import torch.distributions as dists

from radiogalaxies_bnns.inference.models import LeNet, LeNetDrop
import radiogalaxies_bnns.inference.utils as utils
from radiogalaxies_bnns.inference.datamodules import MNISTDataModule, MiraBestDataModule, testloader_mb_uncert
from radiogalaxies_bnns.eval.uncertainty.uncertainty import entropy_MI, overlapping, GMM_logits, calibration

from laplace import Laplace
from netcal.metrics import ECE

def energy_function(logits, T = 1):
    
    # print(-torch.logsumexp(pred_list_mbconf, dim = 2))
    mean_energy = torch.mean(-torch.logsumexp(logits, dim = 2), dim = 0).detach().numpy()
    std_energy = torch.std(-torch.logsumexp(logits, dim = 2), dim = 0).detach().numpy()

    return mean_energy


energy_score_df = pd.read_csv('radiogalaxies_bnns/results/ood/lla_energy_scores.csv', index_col=0)

binwidth = 1#0.10
ylim_lower = 0
ylim_upper = 7

bin_lower = -30 #-80
bin_upper = 2


plt.clf()
plt.figure(dpi=300)
sns.histplot(energy_score_df[['LLA MB Conf', 'LLA Galaxy MNIST', 'LLA MIGHTEE']], binrange = [bin_lower, bin_upper], 
             binwidth = binwidth)
plt.ylim(ylim_lower, ylim_upper)
plt.xlabel('Negative Energy')#Negative
plt.xticks(np.arange(bin_lower, bin_upper, 5))
plt.savefig('radiogalaxies_bnns/results/ood/energy_hist_lla.png')



exit()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_dict, config = utils.parse_config('/share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/inference/nonBayesianCNN/config_cnn.txt')
# seed = 123 #config['training']['seed']
# torch.manual_seed(seed)
path_out = config_dict['output']['path_out']

datamodule = MiraBestDataModule(config_dict, hmc=False)
train_loader = datamodule.train_dataloader()
validation_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()
targets = torch.cat([y for x, y in test_loader], dim=0).to(device)#.numpy()


model = model = LeNet(in_channels=1, output_size=2).to(device)
model_path = path_out + str(8)
model.load_state_dict(torch.load(model_path+'/model', map_location=torch.device(device) ))

@torch.no_grad()
def predict(dataloader, model, laplace=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x.cuda()))
        else:
            py.append(torch.softmax(model(x.cuda()), dim=-1))

    return torch.cat(py)#.cpu().numpy()


# Laplace
la = Laplace(model, 'classification',
             subset_of_weights= 'last_layer',
             hessian_structure='diag')
la.fit(train_loader)
la.optimize_prior_precision(pred_type = 'nn', method='marglik', verbose = True)
# la.optimize_prior_precision(method='marglik')


print("MBCONF")
pred_list_mbconf= utils.get_logits_la(la, test_data_uncert='MBFRConfident', 
                                             device=device, path='./dataMiraBest', 
                                             )
mean_energy_conf = energy_function(pred_list_mbconf)

print("MBUNCERT")
pred_list_mbuncert= utils.get_logits_la(la, test_data_uncert='MBFRUncertain', 
                                               device=device, path='./dataMiraBest',
                                               )
mean_energy_uncert = energy_function(pred_list_mbuncert)

print("MBHYBRID")
pred_list_mbhybrid = utils.get_logits_la(la, test_data_uncert='MBHybrid', 
                                                device=device, path='./dataMiraBest',
                                                )
mean_energy_hybrid = energy_function(pred_list_mbhybrid)


print("GalaxyMNIST")
pred_list_gal_mnist= utils.get_logits_la(la, test_data_uncert='Galaxy_MNIST', 
                                                device=device, path='./dataGalaxyMNISTHighres',
                                                )
mean_energy_galmnist =energy_function(pred_list_gal_mnist)

print("MIGHTEE")
pred_list_mightee = utils.get_logits_la(la, test_data_uncert='mightee', 
                                               device=device, path='./',
                                               )
mean_energy_mightee =energy_function(pred_list_mightee)


s1 = pd.Series(mean_energy_conf, name = 'LLA MB Conf')
s2 = pd.Series(mean_energy_uncert, name = 'LLA MB Uncert')
s3 = pd.Series(mean_energy_hybrid, name = 'LLA MB Hybrid')
s4 = pd.Series(mean_energy_galmnist, name = 'LLA Galaxy MNIST')
s5 = pd.Series(mean_energy_mightee, name = 'LLA MIGHTEE')
energy_score_df = pd.DataFrame([s1, s2, s3, s4, s5]).T



energy_score_df.to_csv('radiogalaxies_bnns/results/ood/lla_energy_scores.csv')

# test_loader, test_data1, data_type, test_data = testloader_mb_uncert(config_dict['output']['test_data'], config_dict['data']['datadir'])


exit()

############################
# UCE Calculation
############################
num_samples = 200

fr1 = 0
fr2 = 0

indices = np.arange(0, len(test_data), 1)
print(indices)

#for each sample in the test set:
for index in indices:

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

    softmax_values = la.predictive_samples(x_test.cuda(), pred_type = 'nn', n_samples = num_samples )
    softmax_ = softmax_values.cpu()

    pred = softmax_.mean(dim=1).argmax(dim=-1).numpy().flatten()
    # pred1 = pred_list.mean(dim=1).argmax(dim=-1)

    y_test_all = np.tile(y_test.detach().numpy().flatten()[0], num_samples)

    # print(pred)
    # print(pred1)
    # print(y_test_all)

    # print(pred != y_test_all)

    errors =  np.mean((pred != y_test_all).astype('uint8'))
    # print(errors)

    softmax = np.array(softmax_).reshape((num_samples, 2))
    logits = pred #np.array(pred) 

    mean_logits = np.mean(logits,axis=0)
    var_logits = np.std(logits,axis=0)
    entropy, mutual_info, entropy_singlepass = entropy_MI(softmax, samples_iter= num_samples)
  

    error_all.append(errors)
    entropy_all.append(entropy/np.log(2))    
    mi_all.append(mutual_info/np.log(2))
    aleat_all.append(entropy_singlepass/np.log(2))


print(error_all)
print(entropy_all)

path = './dropout/mlp/' 

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


uce  = calibration(path, np.array(error_all), np.array(aleat_all), n_bins, x_label = 'average entropy')
print("Average Entropy")
print("uce = ", np.round(uce, 2))
uce_0  = calibration(path, np.array(error_all[fr1_start:fr1_end]), np.array(aleat_all[fr1_start:fr1_end]), n_bins, x_label = 'average entropy') 
print("UCE FRI= ",np.round(uce_0, 2))
uce_1  = calibration(path, np.array(error_all[fr2_start:fr2_end]), np.array(aleat_all[fr2_start:fr2_end]), n_bins, x_label = 'average entropy')
print("UCE FRII = ", np.round(uce_1, 2))
cUCE = (uce_0 + uce_1)/2 
print("cUCE=", np.round(cUCE, 2))  


print("mean and std of error")
print(error_all)
print(np.mean(error_all)*100)
print(np.std(error_all))


exit()



###################################################################################

print(pred)
probs_laplace = predict(test_loader, la, laplace=True)
acc_laplace = ((probs_laplace.argmax(-1) == targets)*1).float().mean()
ece_laplace = ECE(bins=8).measure(probs_laplace.cpu().numpy(), targets.cpu().numpy())
nll_laplace = -dists.Categorical(probs_laplace).log_prob(targets).mean()

print(f'[Diag Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}')




exit()
print('Dropout')

#Non-Bayesian CNN if dropout_rate = 0
model = LeNetDrop(in_channels = 1, output_size =  2, dropout_rate = 0.5).to(device) #MLP(150, 200, 10)


model.load_state_dict(torch.load('./dropout/exp3/model', map_location=torch.device('cpu') ))

probs_map = predict(test_loader, model, laplace=False)
print((probs_map.argmax(-1) == targets)*1)
acc_map = ((probs_map.argmax(-1) == targets)*1).float().mean()
ece_map = ECE(bins=8).measure(probs_map.cpu().numpy(), targets.cpu().numpy())
nll_map = -dists.Categorical(probs_map).log_prob(targets).mean()
print(f'[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}')


# Laplace
la = Laplace(model, 'classification',
             subset_of_weights= 'last_layer',
             hessian_structure='diag')
la.fit(train_loader)
la.optimize_prior_precision(method='marglik')


probs_laplace = predict(test_loader, la, laplace=True)
acc_laplace = ((probs_laplace.argmax(-1) == targets)*1).float().mean()
ece_laplace = ECE(bins=8).measure(probs_laplace.cpu().numpy(), targets.cpu().numpy())
nll_laplace = -dists.Categorical(probs_laplace).log_prob(targets).mean()

print(f'[Diag Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}')

