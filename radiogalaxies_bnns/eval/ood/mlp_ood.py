import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
import scipy
import pandas as pd
import seaborn as sns

from radiogalaxies_bnns.inference.models import LeNet, LeNetDrop
import radiogalaxies_bnns.inference.utils as utils
from radiogalaxies_bnns.inference.datamodules import MNISTDataModule, MiraBestDataModule, testloader_mb_uncert
from radiogalaxies_bnns.eval.uncertainty.uncertainty import entropy_MI, overlapping, GMM_logits, calibration


def energy_function(logits, T = 1):
    
    # print(-torch.logsumexp(pred_list_mbconf, dim = 2))
    mean_energy = torch.mean(-torch.logsumexp(logits, dim = 2), dim = 0).detach().numpy()
    std_energy = torch.std(-torch.logsumexp(logits, dim = 2), dim = 0).detach().numpy()

    return mean_energy

def energy_function_mlp(logits, T = 1):
    
    # print(-torch.logsumexp(pred_list_mbconf, dim = 2))
    mean_energy = -torch.logsumexp(logits, dim = 1).detach().numpy()
    # std_energy = torch.std(-torch.logsumexp(logits, dim = 2), dim = 0).detach().numpy()

    return mean_energy



energy_score_df = pd.read_csv('radiogalaxies_bnns/results/ood/mlp_energy_scores.csv', index_col=0)

binwidth = 1#0.10
ylim_lower = 0
ylim_upper = 7

bin_lower = -30 #-80
bin_upper = 2

plt.clf()
plt.figure(dpi=300)
sns.histplot(energy_score_df[['MLP MB Conf', 'MLP Galaxy MNIST', 'MLP MIGHTEE']], binrange = [bin_lower, bin_upper], 
             binwidth = binwidth)
plt.ylim(ylim_lower, ylim_upper)
plt.xlabel('Negative Energy')#Negative
plt.xticks(np.arange(bin_lower, bin_upper, 5))
plt.savefig('radiogalaxies_bnns/results/ood/energy_hist_mlp.png')



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
# print('TEST DATASET')
# print(config_dict['output']['test_data'])
# test_loader, test_data1, data_type, test_data = testloader_mb_uncert(config_dict['output']['test_data'], config_dict['data']['datadir'])


model = LeNet(in_channels=1, output_size=2).to(device)
model_path = path_out + str(8)
model.load_state_dict(torch.load(model_path+'/model', map_location=torch.device(device) ))

# test_error = eval(test_loader)
# print(test_error)


print("MBCONF")
pred_list_mbconf= utils.get_logits_mlp(model, test_data_uncert='MBFRConfident', device=device, path='./dataMiraBest')
mean_energy_conf = energy_function_mlp(pred_list_mbconf)


print("MBUNCERT")
pred_list_mbuncert= utils.get_logits_mlp(model, test_data_uncert='MBFRUncertain', device=device, path='./dataMiraBest')
mean_energy_uncert = energy_function_mlp(pred_list_mbuncert)

print("MBHYBRID")
pred_list_mbhybrid = utils.get_logits_mlp(model, test_data_uncert='MBHybrid', device=device, path='./dataMiraBest')
mean_energy_hybrid = energy_function_mlp(pred_list_mbhybrid)


print("GalaxyMNIST")
pred_list_gal_mnist= utils.get_logits_mlp(model, test_data_uncert='Galaxy_MNIST', device=device, path='./dataGalaxyMNISTHighres')
mean_energy_galmnist =energy_function_mlp(pred_list_gal_mnist)

print("MIGHTEE")
pred_list_mightee = utils.get_logits_mlp(model, test_data_uncert='mightee', device=device, path='./')
mean_energy_mightee =energy_function_mlp(pred_list_mightee)


s1 = pd.Series(mean_energy_conf, name = 'MLP MB Conf')
s2 = pd.Series(mean_energy_uncert, name = 'MLP MB Uncert')
s3 = pd.Series(mean_energy_hybrid, name = 'MLP MB Hybrid')
s4 = pd.Series(mean_energy_galmnist, name = 'MLP Galaxy MNIST')
s5 = pd.Series(mean_energy_mightee, name = 'MLP MIGHTEE')
energy_score_df = pd.DataFrame([s1, s2, s3, s4, s5]).T



energy_score_df.to_csv('radiogalaxies_bnns/results/ood/mlp_energy_scores.csv')



exit()


