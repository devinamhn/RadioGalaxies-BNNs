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


def entropy_MI(softmax, samples_iter):

    class_0 = softmax[:, :,0]
    class_1 = softmax[:, :,1]
    #biased estimator of predictive entropy, bias will reduce as samples_iter is increased
    entropy = -((np.sum(class_0, axis = 1)/samples_iter) * np.log(np.sum(class_0, axis = 1)/samples_iter) + (np.sum(class_1, axis = 1)/samples_iter) * np.log(np.sum(class_1, axis = 1)/samples_iter))
    
    mutual_info = entropy + np.sum(class_0*np.log(class_0), axis = 1)/samples_iter +  np.sum(class_1*np.log(class_1), axis = 1)/samples_iter
    
    return entropy, mutual_info

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
entropy_all = []
mi_all = []
rgz_id_all = []
for i, (x_test_rgz, y_test_rgz) in enumerate(test_loader):
    n_gal = x_test_rgz.data.shape[0]
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
    softmax_ = F.softmax(pred_list_rgz, dim=-1).cpu()
    softmax = np.array(softmax_).reshape((n_gal, len(params_hmc), 2))
    entropy, mutual_info= entropy_MI(softmax, samples_iter= len(params_hmc))
    
    rgz_id_all.append(rgz_id)
    entropy_all.append(entropy)
    mi_all.append(mutual_info)


id_df = pd.Series(np.concatenate(rgz_id_all), name = 'id')
entropy_df = pd.Series(np.concatenate(entropy_all), name = 'pred entropy')
mi_df = pd.Series(np.concatenate(mi_all), name = 'mutual info')

entropy_df = pd.DataFrame([id_df, entropy_df, mi_df]).T


entropy_df.to_csv('radiogalaxies_bnns/results/ood/hmc_entropy_rgz.csv')

entropy_df = pd.read_csv('radiogalaxies_bnns/results/ood/hmc_entropy_rgz.csv', index_col=0)

binwidth = 0.01#0.10
# ylim_lower = 0
# ylim_upper = 7

# bin_lower = -80
# bin_upper = 10

plt.clf()
plt.figure(dpi=300)
sns.histplot(entropy_df[['pred entropy']], 
            #  binrange = [bin_lower, bin_upper], 
             binwidth = binwidth)
# plt.ylim(ylim_lower, ylim_upper)
plt.xlabel('Predictive entropy')#Negative
# plt.xticks(np.arange(-80, 10, 10))
plt.savefig('radiogalaxies_bnns/results/ood/entropy_hist_hmc_rgz.png')

plt.clf()
plt.figure(dpi=300)
sns.histplot(entropy_df[['mutual info']], 
            #  binrange = [bin_lower, bin_upper], 
             binwidth = binwidth)
# plt.ylim(ylim_lower, ylim_upper)
plt.xlabel('Mutual information')#Negative
# plt.xticks(np.arange(-80, 10, 10))
plt.savefig('radiogalaxies_bnns/results/ood/mutualinfo_hist_hmc_rgz_std.png')


entropy_df = pd.read_csv('radiogalaxies_bnns/results/ood/hmc_entropy_rgz.csv', index_col=0)
energy_score_df = pd.read_csv('radiogalaxies_bnns/results/ood/hmc_energy_scores_rgz_sigma_std.csv', index_col=0)
rgz_ids = pd.read_csv('radiogalaxies_bnns/eval/rgz/rgz_inference/static_rgz_flat_2019-05-06_full.csv', index_col=0)
rgz_ids.rename(columns={'rgz_name':'id'}, inplace = True)
merged_df = energy_score_df.merge(entropy_df[['id', 'pred entropy', 'mutual info']], how ='left', on ='id')
all_df = merged_df.merge(rgz_ids[['id', 'radio.ra', 'radio.dec']], how ='left', on ='id')
all_df.drop_duplicates(subset='id', inplace = True)
all_df.to_csv('radiogalaxies_bnns/results/ood/hmc_rgz.csv')
