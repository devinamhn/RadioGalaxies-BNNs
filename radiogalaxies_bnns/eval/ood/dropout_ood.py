import torch
from models import LeNetDrop, LeNet
import utils
from datamodules import MNISTDataModule, MiraBestDataModule, testloader_mb_uncert
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
from uncertainty import entropy_MI, overlapping, GMM_logits
import scipy
import pandas as pd
import seaborn as sns

def enable_dropout(m):
  for each_module in m.modules():
    if each_module.__class__.__name__.startswith('Dropout'):
      each_module.train()



def eval(test_loader):

    model.train(False)
    running_error = 0.0
    enable_dropout(model)
    for i, (x_test, y_test) in enumerate(test_loader):
        x_test, y_test = x_test.to(device), y_test.to(device)
        # forward + backward + optimize
        outputs = model(x_test)
        # loss = criterion(outputs, y_val)
        # loss.backward()
        # optimizer.step()
        pred = F.softmax(outputs, dim = -1).argmax(dim=-1)
        error = (pred!= y_test).to(torch.float32).mean()
        # print statistics
        # running_loss += loss.item()
        running_error += error

    # print('batch size, i', i+1)
    return running_error/(i+1)

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





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_dict, config = utils.parse_config('config_mb_dropout.txt')
seed = 123 #config['training']['seed']
torch.manual_seed(seed)
path_out = config_dict['output']['path_out']


datamodule = MiraBestDataModule(config_dict, hmc=False)
train_loader = datamodule.train_dataloader()
validation_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()

# model = LeNetDrop(in_channels = 1, output_size =  2, dropout_rate = 0.5 ).to(device) #MLP(150, 200, 10)
model = LeNetDrop(in_channels=1, output_size=2, dropout_rate = 0)

#load trained model
model.load_state_dict(torch.load('./dropout/mlp/model', map_location=torch.device('cpu') )) #exp3

pred_list_mbconf = utils.get_logits_mlp(model, test_data_uncert='MBFRConfident', device=device, path='./dataMiraBest')
mean_energy_conf = energy_function_mlp(pred_list_mbconf)


pred_list_gal_mnist = utils.get_logits_mlp(model, test_data_uncert='Galaxy_MNIST', device=device, path='./dataGalaxyMNISTHighres')
mean_energy_galmnist =energy_function_mlp(pred_list_gal_mnist)


energy_score_df = pd.DataFrame({'CNN MB Conf': mean_energy_conf, 
                                'CNN Galaxy MNIST': mean_energy_galmnist,
                                }) #, mean_enerygy_galmnist.reshape(100, 1)])
# energy_score_df['dataset'] = dataset
print(energy_score_df)
energy_score_df.to_csv('./results/ood/mlp_energy_scores.csv')

sns.histplot(energy_score_df) #, hue = 'dataset')
plt.xlabel('Negative Energy')
plt.savefig('./energy_hist_mlp.png')







exit()
mean_energy_conf = energy_function_mlp(pred_list_mbconf)
print(mean_energy_conf)
# i = 0
# for name, param in model.named_parameters():
#   print(torch.flatten(param).shape)
#   cov = torch.cov(param)
#   print(cov)

#   # # samples0_df = pd.DataFrame(samples_corner0, index = ['w1', 'w2', 'w3', 'w4', 'w5'])
#   # plt.figure(dpi = 200)
#   sns.heatmap(cov.detach().numpy())
#   plt.title('Layer' + str(i))
#   plt.savefig('./heatmap_mlp_l' +str(i) + '.png')

exit()
test_error = eval(test_loader)
print(test_error)



pred_list_mbconf = utils.get_logits(model, test_data_uncert='MBFRConfident', device=device, path='./dataMiraBest')
print(pred_list_mbconf)
print(pred_list_mbconf.shape)
mean_energy_conf = energy_function(pred_list_mbconf)
print(mean_energy_conf)

pred_list_gal_mnist = utils.get_logits(model, test_data_uncert='Galaxy_MNIST', device=device, path='./dataGalaxyMNISTHighres')
print(pred_list_mbconf)
print(pred_list_mbconf.shape)


mean_energy_galmnist =energy_function(pred_list_gal_mnist)


energy_score_df = pd.DataFrame({'MB Conf': mean_energy_conf, 
                                'Galaxy MNIST': mean_energy_galmnist,
                                }) #, mean_enerygy_galmnist.reshape(100, 1)])
# energy_score_df['dataset'] = dataset
print(energy_score_df)
energy_score_df.to_csv('./results/ood/dropout_energy_scores.csv')

sns.kdeplot(energy_score_df) #, hue = 'dataset')
plt.xlabel('Negative Energy')
plt.savefig('./energy_kde.png')