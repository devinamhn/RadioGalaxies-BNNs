import torch
import hamiltorch
from radiogalaxies_bnns.inference.datamodules import MiraBestDataModule
import radiogalaxies_bnns.inference.utils as utils
from radiogalaxies_bnns.inference.models import LeNet
import sys

paths = utils.Path_Handler()._dict()
config_dict, config = utils.parse_config(paths['inference']/ 'hmc'/ 'config_mb.txt')
jobid = int(sys.argv[1])
print(jobid)

hamiltorch.set_random_seed(config_dict['training']['seed']) #+ jobid)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LeNet(1, 2)

datamodule = MiraBestDataModule(config_dict, hmc=True)
train_loader = datamodule.train_dataloader()
validation_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()

step_size = 0.0001
num_samples = 200000
L = 50
tau_out = 1.
normalizing_const = 1.
burn = 0

tau_list = []
tau = 100. #std = 0.1
for w in model.parameters():
    tau_list.append(tau)
tau_list = torch.tensor(tau_list).to(device)

params_init = hamiltorch.util.flatten(model).to(device).clone()

batch = 1
num_batches = 1
for batch, (x_train, y_train) in enumerate(train_loader):
    x_train, y_train = x_train.to(device), y_train.to(device)

checkpt = 1000
n_checkpt = int(num_samples/checkpt)

path = './results/inits/mb_out_' + str(jobid) + '/'

for i in range(n_checkpt):
    if(i!=0):
        #load in the previous checkpoint
        params_init = torch.load(path + 'params_hmc_checkpt.pt')
    params_hmc = hamiltorch.sample_model(model, x_train, y_train, params_init=params_init, 
                                         model_loss='multi_class_linear_output', num_samples=checkpt, burn = burn,
                                         step_size=step_size, num_steps_per_sample=L,tau_out=tau_out,
                                         tau_list=tau_list, normalizing_const=normalizing_const)
    
    torch.save(params_hmc, path + 'params_hmc_'+ str(i) + '.pt')
    torch.save(params_hmc[checkpt-1][:], path + 'params_hmc_checkpt.pt')

exit()
