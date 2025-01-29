from radiogalaxies_bnns.eval.rgz.rgz_inference.rgz_datamodules import RGZ_DataModule
from radiogalaxies_bnns.inference.utils import Path_Handler
from radiogalaxies_bnns.datasets.rgz108k import RGZ108k 
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F

from matplotlib import pyplot as plt 
import torch
from radiogalaxies_bnns.inference.models import LeNet
import hamiltorch
import pandas as pd
import numpy as np
import seaborn as sns

def energy_function(logits, T = 1):
    
    mean_energy = torch.mean(-torch.logsumexp(logits, dim = 2), dim = 0).cpu().detach().numpy()
    std_energy = torch.std(-torch.logsumexp(logits, dim = 2), dim = 0).cpu().detach().numpy()

    return mean_energy, std_energy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

paths = Path_Handler()._dict()

transform = T.Compose(
    [
        # T.CenterCrop(70),
        T.ToTensor(),
        T.Normalize((0.008008896,), (0.05303395,)),
    ]
)
# Load in RGZ dataset
test_dataset = RGZ108k(paths["rgz"], train=True, transform=transform)


batch_size = 1000
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
mean_energy_rgz_all = []
std_energy_rgz_all = []
rgz_id_all = []
las_all = []

for i, (x_test_rgz, y_test_rgz) in enumerate(test_loader):
    x_test_rgz = x_test_rgz.data.to(device)
    rgz_id = y_test_rgz['id']
    las = y_test_rgz['size']
    y_test_rgz = torch.zeros(x_test_rgz.data.shape[0]).to(device)

    model = LeNet(1, 2)
    path = '/share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/results/inits/'
    params_hmc = torch.load(path+'/thin_chain'+str(0), map_location=torch.device(device)) 

    burn_in = 0 #20
    params_hmc = params_hmc[burn_in:]

    tau_list = []
    tau = 100. 

    for w in model.parameters():
        tau_list.append(tau)
    tau_list = torch.tensor(tau_list).to(device)

    temp = 1   
    pred_list_rgz, log_prob_list_rgz = hamiltorch.predict_model(model, x = x_test_rgz.data, y = y_test_rgz, 
                                                        samples=params_hmc, model_loss='multi_class_linear_output', 
                                                        tau_out=1., tau_list=tau_list)#, temp = temp)

    mean_energy_rgz, std_energy_rgz = energy_function(pred_list_rgz)
    mean_energy_rgz_all.append(mean_energy_rgz)
    std_energy_rgz_all.append(std_energy_rgz)
    rgz_id_all.append(rgz_id)
    las_all.append(las)

id_df = pd.Series(np.concatenate(rgz_id_all), name = 'id')
las_df = pd.Series(np.concatenate(las_all), name = 'las')
mean_energy_df = pd.Series(np.concatenate(mean_energy_rgz_all), name = 'HMC RGZ')
std_energy_df = pd.Series(np.concatenate(std_energy_rgz_all), name = 'HMC RGZ std')

energy_score_df = pd.DataFrame([id_df, las_df, mean_energy_df, std_energy_df]).T
# print(energy_score_df)
# exit()

energy_score_df.to_csv('radiogalaxies_bnns/results/ood/hmc_energy_scores_rgz.csv')

energy_score_df = pd.read_csv('radiogalaxies_bnns/results/ood/hmc_energy_scores_rgz.csv', index_col=0)

binwidth = 0.1#0.10
# ylim_lower = 0
# ylim_upper = 7

# bin_lower = -80
# bin_upper = 10

plt.clf()
plt.figure(dpi=300)
sns.histplot(energy_score_df[['HMC RGZ']], 
            #  binrange = [bin_lower, bin_upper], 
             binwidth = binwidth)
# plt.ylim(ylim_lower, ylim_upper)
plt.xlabel('Mean Negative Energy')#Negative
# plt.xticks(np.arange(-80, 10, 10))
plt.savefig('radiogalaxies_bnns/results/ood/energy_hist_hmc_rgz.png')

plt.clf()
plt.figure(dpi=300)
sns.histplot(energy_score_df[['HMC RGZ std']], 
            #  binrange = [bin_lower, bin_upper], 
             binwidth = binwidth)
# plt.ylim(ylim_lower, ylim_upper)
plt.xlabel('Std of Negative Energy')#Negative
# plt.xticks(np.arange(-80, 10, 10))
plt.savefig('radiogalaxies_bnns/results/ood/energy_hist_hmc_rgz_std.png')


