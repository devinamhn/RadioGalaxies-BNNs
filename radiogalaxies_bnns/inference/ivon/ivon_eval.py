import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from radiogalaxies_bnns.inference.ivon._ivon import IVON

import radiogalaxies_bnns.inference.utils as utils
import radiogalaxies_bnns.inference.ivon.ivon_utils as ivon_utils

from radiogalaxies_bnns.inference.models import LeNet
from radiogalaxies_bnns.inference.datamodules import MNISTDataModule, MiraBestDataModule, testloader_mb_uncert


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


mean_expected_error, std_expected_error, uce_pe, uce_mi, uce_ae = np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)


for i in range(10):
    model_path = path_out + str(i+1)
    model = LeNet(in_channels = 1, output_size =  2 ).to(device) #MLP(150, 200, 10)
    optimizer = IVON(model.parameters(), lr=lr, ess=ess, weight_decay=weight_decay, hess_init=h0,
                        beta1=momentum, beta2=beta_2)
    model.load_state_dict(torch.load(model_path+'/model', map_location=torch.device(device) ))

    mean_expected_error[i], std_expected_error[i], uce_pe[i], uce_mi[i], uce_ae[i] = ivon_utils.calibration_test(path_out, model, optimizer, test_data_uncert = 'MBFRConfident', device=device, path=config_dict['data']['datadir'], test = True, samples_iter=mc_samples)

print("Mean of error over seeds", mean_expected_error.mean())
print("Std of error over seeds", mean_expected_error.std())

print("Mean of std over seeds", std_expected_error.mean())
print("Std of std over seeds", std_expected_error.std())


print('Mean of uce pe', uce_pe.mean())
print('Std of uce pe', uce_pe.std())

print('Mean of uce mi', uce_mi.mean())
print('Std of uce mi', uce_mi.std())

print('Mean of uce ae', uce_ae.mean())
print('Std of uce ae', uce_ae.std())