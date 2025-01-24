from radiogalaxies_bnns.eval.rgz.rgz_inference.rgz_datamodules import RGZ_DataModule
from radiogalaxies_bnns.inference.utils import Path_Handler
from radiogalaxies_bnns.datasets.rgz108k import RGZ108k 
from torch.utils.data import DataLoader
import torchvision.transforms as T
from matplotlib import pyplot as plt 
import torch
from radiogalaxies_bnns.inference.models import LeNet
import hamiltorch
import pandas as pd
import numpy as np
import seaborn as sns

def energy_function(logits, T = 1):
    
    # print(-torch.logsumexp(pred_list_mbconf, dim = 2))
    mean_energy = torch.mean(-torch.logsumexp(logits, dim = 2), dim = 0).cpu().detach().numpy()
    std_energy = torch.std(-torch.logsumexp(logits, dim = 2), dim = 0).cpu().detach().numpy()

    return mean_energy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

paths = Path_Handler()._dict()

# datamodule = RGZ_DataModule(
#     path=paths["rgz"],
#     batch_size= 1024, #config["data"]["batch_size"],
#     center_crop= 70, #config["augmentations"]["center_crop"],
#     random_crop= [0.8, 1], #config["augmentations"]["random_crop"],
#     s=0.5, #config["augmentations"]["s"],
#     p_blur= 0.1, # config["augmentations"]["p_blur"],
#     flip= True, #config["augmentations"]["flip"],
#     rotation= True, #config["augmentations"]["rotation"],
#     cut_threshold= 20, #config["data"]["cut_threshold"],
#     prefetch_factor= 30, #config["dataloading"]["prefetch_factor"],
#     num_workers= 8 #config["dataloading"]["num_workers"],
# )

# print(datamodule)

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
test_loader = DataLoader(test_dataset, batch_size= 100, shuffle=False)
mean_energy_rgz_all = []
for i, (x_test_rgz, y_test_rgz) in enumerate(test_loader):
    x_test_rgz = x_test_rgz.data.to(device)
    # print(x_test_rgz.data.shape,x_test_rgz.data.shape[0] )
    y_test_rgz = torch.zeros(x_test_rgz.data.shape[0] ).to(device)
    # if(i == 2):
    #     break


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


    mean_energy_rgz = energy_function(pred_list_rgz)
    mean_energy_rgz_all.append(mean_energy_rgz)
    # print(mean_energy_rgz)
    # print(mean_energy_rgz_all)

s6 = pd.Series(np.concatenate(mean_energy_rgz_all), name = 'HMC RGZ')
# print(s6)
energy_score_df = pd.DataFrame([s6]).T
# print(energy_score_df)


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
plt.xlabel('Negative Energy')#Negative
# plt.xticks(np.arange(-80, 10, 10))
plt.savefig('radiogalaxies_bnns/results/ood/energy_hist_hmc_rgz.png')




