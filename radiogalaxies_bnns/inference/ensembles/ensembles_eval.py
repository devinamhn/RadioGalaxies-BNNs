import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


import radiogalaxies_bnns.inference.utils as utils
import radiogalaxies_bnns.inference.ensembles.ensembles_utils as ensemble_utils

from radiogalaxies_bnns.inference.models import LeNet
from radiogalaxies_bnns.inference.datamodules import MNISTDataModule, MiraBestDataModule, testloader_mb_uncert


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_dict, config = utils.parse_config('/share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/inference/ensembles/config_ensembles.txt')
jobid = int(sys.argv[1])
# seed = 123 + jobid #config['training']['seed']
# torch.manual_seed(seed)
path_out = config_dict['output']['path_out']
n_ensembles = 5

datamodule = MiraBestDataModule(config_dict, hmc=False)
train_loader = datamodule.train_dataloader()
validation_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()

model = LeNet(in_channels = 1, output_size =  2).to(device) #MLP(150, 200, 10)
# model_path = path_out+'model'+str(jobid)

for i in range(n_ensembles):
    model.load_state_dict(torch.load('/share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/results/ensembles/model' + str(i)))
    test_error = ensemble_utils.eval(model, test_loader, device)
    print('Test error: {} %'.format(test_error*100))

# test_err = torch.zeros(200)
# for i in range(200):
#     test_err[i] = dropout_utils.eval(model, test_loader, device)

# err_mean = torch.mean(test_err)
# err_std = torch.std(test_err)
# print('Test error mean: {} % '.format(err_mean*100))
# print('Test error std: {} % '.format(err_std))

