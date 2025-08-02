import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from radiogalaxies_bnns.inference.models import LeNet, LeNetDrop
import radiogalaxies_bnns.inference.utils as utils
import radiogalaxies_bnns.inference.ivon.ivon_utils as ivon_utils
from radiogalaxies_bnns.inference.ivon._ivon import IVON

from radiogalaxies_bnns.inference.datamodules import MNISTDataModule, MiraBestDataModule, testloader_mb_uncert
from radiogalaxies_bnns.eval.uncertainty.uncertainty import entropy_MI, overlapping, GMM_logits, calibration


def energy_function(logits, T = 1):
    
    # print(-torch.logsumexp(pred_list_mbconf, dim = 2))
    mean_energy = torch.mean(-torch.logsumexp(logits, dim = 2), dim = 0).detach().numpy()
    std_energy = torch.std(-torch.logsumexp(logits, dim = 2), dim = 0).detach().numpy()

    return mean_energy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_dict, config = utils.parse_config('/share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/inference/ivon/config_ivon.txt')
seed = config_dict['training']['seed']
torch.manual_seed(seed)
path_out = config_dict['output']['path_out']

lr = config_dict['training']['learning_rate']
weight_decay = config_dict['training']['weight_decay']
momentum = 0.9
beta_2 = 1 - 1e-6
h0 = 0.5
ess = 584 * 100
mc_samples = 200 #mc_samples

datamodule = MiraBestDataModule(config_dict, hmc=False)
train_loader = datamodule.train_dataloader()
validation_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()


model = LeNet(in_channels=1, output_size=2).to(device)
model_path = path_out + str(8)
model.load_state_dict(torch.load(model_path+'/model', map_location=torch.device(device) ))

optimizer = IVON(model.parameters(), lr=lr, ess=ess, weight_decay=weight_decay, hess_init=h0,
                    beta1=momentum, beta2=beta_2)


print("MBCONF")
pred_list_mbconf= ivon_utils.get_logits(model, test_data_uncert='MBFRConfident', device=device, 
path='/share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/data/dataMiraBest' , optimizer = optimizer)
# print(pred_list_mbconf)
# print(pred_list_mbconf.shape)
mean_energy_conf = energy_function(pred_list_mbconf)
# print(mean_energy_conf)

print("MBUNCERT")
pred_list_mbuncert= ivon_utils.get_logits(model, test_data_uncert='MBFRUncertain', device=device, 
path='/share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/data/dataMiraBest', optimizer = optimizer)
# print(pred_list_mbconf)
# print(pred_list_mbconf.shape)
mean_energy_uncert = energy_function(pred_list_mbuncert)
# print(mean_energy_conf)

print("MBHYBRID")
pred_list_mbhybrid = ivon_utils.get_logits(model, test_data_uncert='MBHybrid', device=device, 
path='/share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/data/dataMiraBest', optimizer = optimizer)
# print(pred_list_mbconf)
# print(pred_list_mbconf.shape)
mean_energy_hybrid = energy_function(pred_list_mbhybrid)
# print(mean_energy_conf)


print("GalaxyMNIST")
pred_list_gal_mnist= ivon_utils.get_logits(model, test_data_uncert='Galaxy_MNIST', device=device, path='./dataGalaxyMNISTHighres', optimizer = optimizer)
# print(pred_list_gal_mnist)
# print(pred_list_gal_mnist.shape)
mean_energy_galmnist =energy_function(pred_list_gal_mnist)
# print(mean_energy_galmnist)

print("MIGHTEE")
pred_list_mightee = ivon_utils.get_logits(model, test_data_uncert='mightee', device=device, path='./', optimizer = optimizer)
# print(pred_list_mightee)
# print(pred_list_mightee.shape)
mean_energy_mightee =energy_function(pred_list_mightee)
# print(mean_energy_mightee)

# pred_list_mbconf = utils.get_logits(model, test_data_uncert='MBFRConfident', device=device, path='./dataMiraBest')
# print(pred_list_mbconf)
# print(pred_list_mbconf.shape)
# mean_energy_conf = energy_function(pred_list_mbconf)
# print(mean_energy_conf)

# pred_list_gal_mnist = utils.get_logits(model, test_data_uncert='Galaxy_MNIST', device=device, path='./dataGalaxyMNISTHighres')
# print(pred_list_mbconf)
# print(pred_list_mbconf.shape)
# mean_energy_galmnist =energy_function(pred_list_gal_mnist)



s1 = pd.Series(mean_energy_conf, name = 'iVON MB Conf')
s2 = pd.Series(mean_energy_uncert, name = 'iVON MB Uncert')
s3 = pd.Series(mean_energy_hybrid, name = 'iVON MB Hybrid')
s4 = pd.Series(mean_energy_galmnist, name = 'iVON Galaxy MNIST')
s5 = pd.Series(mean_energy_mightee, name = 'iVON MIGHTEE')
energy_score_df = pd.DataFrame([s1, s2, s3, s4, s5]).T



energy_score_df.to_csv('radiogalaxies_bnns/results/ood/ivon_energy_scores.csv')



binwidth = 1#0.10
ylim_lower = 0
ylim_upper = 7

bin_lower = -30 #-80
bin_upper = 2

plt.clf()
plt.figure(dpi=300)
sns.histplot(energy_score_df[['iVON MB Conf', 'iVON Galaxy MNIST', 'iVON MIGHTEE']], binrange = [bin_lower, bin_upper], 
             binwidth = binwidth)
plt.ylim(ylim_lower, ylim_upper)
plt.xlabel('Negative Energy')#Negative
plt.xticks(np.arange(bin_lower, bin_upper, 5))
plt.savefig('radiogalaxies_bnns/results/ood/energy_hist_ivon.png')



exit()
