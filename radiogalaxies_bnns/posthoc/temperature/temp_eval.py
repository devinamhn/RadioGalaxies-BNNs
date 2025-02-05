import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb


import radiogalaxies_bnns.inference.utils as utils
import radiogalaxies_bnns.inference.nonBayesianCNN.utils as cnn_utils

from radiogalaxies_bnns.inference.models import LeNet
from radiogalaxies_bnns.inference.datamodules import MNISTDataModule, MiraBestDataModule, testloader_mb_uncert
from radiogalaxies_bnns.eval.uncertainty.uncertainty import entropy_MI, overlapping, GMM_logits, calibration
from radiogalaxies_bnns.posthoc.temperature.temp_scaling import ModelWithTemperature


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_dict, config = utils.parse_config('/share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/inference/nonBayesianCNN/config_cnn.txt')
optimal_temp = np.zeros(10)
original_nll, calibrated_nll, original_ece, calibrated_ece = np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)


for i in range(10):
    seed = config_dict['training']['seed'] + (i+1)
    seed_dataset = config_dict['training']['seed_dataset'] + (i+1)

    torch.manual_seed(seed)
    path_out = config_dict['output']['path_out']
    criterion = torch.nn.CrossEntropyLoss()


    datamodule = MiraBestDataModule(config_dict, hmc=False, seed_dataset=seed_dataset)
    train_loader = datamodule.train_dataloader()
    #must use the same validation loader that was used to train the model
    validation_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    model = LeNet(in_channels = 1, output_size =  2).to(device)
    model_path = path_out+ str(i+1) +'/model' 

    #get uncalibrated model
    model.load_state_dict(torch.load(model_path))
    scaled_model = ModelWithTemperature(model)

    optimal_temp[i], original_nll[i], original_ece[i], calibrated_nll[i], calibrated_ece[i] = scaled_model.set_temperature(validation_loader)

ensemble_n = np.linspace(1, 10, 10)
print(optimal_temp)

plt.figure(dpi = 200)
plt.scatter(ensemble_n, optimal_temp)
plt.title('Optimal temp')
plt.savefig('optimal_temp')
plt.clf()

plt.scatter(ensemble_n, original_nll)
plt.scatter(ensemble_n, calibrated_nll)
plt.title('NLL')
plt.savefig('nll')
plt.clf()


plt.scatter(ensemble_n, original_ece)
plt.scatter(ensemble_n, calibrated_ece)
plt.title('ECE')
plt.savefig('ece')

exit()

print('TEST DATASET')
print(config_dict['output']['test_data'])
test_loader, test_data1, data_type, test_data = testloader_mb_uncert(config_dict['output']['test_data'], config_dict['data']['datadir'])



n_ensembles = 10



avg_error_mean = []

error_all = []
entropy_all = []
mi_all = []
aleat_all = []
loss_all = [] 

fr1 = 0
fr2 = 0
indices = np.arange(0, len(test_data), 1)

for index in indices:

    x = torch.unsqueeze(test_data[index][0].clone().detach(), 0)
    y = torch.unsqueeze(torch.tensor(test_data[index][1]),0)

    target = y.detach().numpy().flatten()[0]

    if(target == 0):
        fr1+= 1
    elif(target==1):
        fr2+= 1
    
    output_ = []
    logits_=[]
    prediction=[]
    y_test_all = []
    errors = []

    with torch.no_grad():
        model.train(False)
        for j in range(n_ensembles):
            x_test, y_test = x.to(device), y.to(device)

            model_path = path_out+ str(j+1) +'/model'# +'/model' #
            model.load_state_dict(torch.load(model_path))
            
            outputs = model(x_test)
            softmax = F.softmax(outputs, dim = -1)
            pred = softmax.argmax(dim=-1)

            # loss = criterion(outputs, torch.tile(y_test[k], (n_ensembles, 1)).flatten())

            output_.append(softmax.cpu().detach().numpy().flatten())
            logits_.append(outputs.cpu().detach().numpy().flatten())
            
            # predict = pred.mean(dim=0).argmax(dim=-1)

            prediction.append(pred.cpu().detach().numpy().flatten()[0])
            y_test_all.append(y_test.cpu().detach().numpy().flatten()[0])
    
    softmax = np.array(output_)#.cpu().detach().numpy())
    y_logits = np.array(logits_)#.cpu().detach().numpy())

    sorted_softmax_values, lower_index, upper_index, mean_samples = credible_interval(softmax[:, 0].flatten(), 1) #0.64
    upper_index = upper_index - 1
    # print("90% credible interval for FRI class softmax", sorted_softmax_values[lower_index], sorted_softmax_values[upper_index])
    
    sorted_softmax_values_fr1 = sorted_softmax_values[lower_index:upper_index]
    sorted_softmax_values_fr2 = 1 - sorted_softmax_values[lower_index:upper_index]

    softmax_mean = np.vstack((mean_samples, 1-mean_samples)).T
    softmax_credible = np.vstack((sorted_softmax_values_fr1, sorted_softmax_values_fr2)).T

    entropy, mutual_info, entropy_singlepass = entropy_MI(softmax_credible, 
                                                          samples_iter= len(softmax_credible[:,0]))



    pred_mean = np.argmax(softmax_mean, axis = 1)
    error_mean = (pred_mean != target)*1


    pred = np.argmax(softmax_credible, axis = 1)

    y_test_all = np.tile(target, len(softmax_credible[:,0]))

    errors =  np.mean((pred != y_test_all).astype('uint8'))

    avg_error_mean.append(error_mean)
    error_all.append(errors)
    entropy_all.append(entropy/np.log(2))    
    mi_all.append(mutual_info/np.log(2))
    aleat_all.append(entropy_singlepass/np.log(2))
    # loss_all.append(loss.item())


print("mean and std of error")
print(error_all)
print("mean", np.round(np.mean(error_all)*100, 3))
print("std", np.round(np.std(error_all), 3 ))

print("Average of expected error")
print((np.array(avg_error_mean)).mean()*100)
print((np.array(avg_error_mean)).std())



# print("Average CE Loss")
# # print(loss_all)
# print("mean", np.round(np.mean(loss_all), 3))
# print("std", np.round(np.std(loss_all), 3 ))


fr1_start = 0 #{0}
fr1_end = fr1 #{49, 68}
fr2_start = fr1 #49, 68
fr2_end = len(y_test) #len(val_indices) #{104, 145}
    
n_bins = 8
print('BINS', 8)
path = './'
uce  = calibration(path, np.array(error_all), np.array(entropy_all), n_bins, x_label = 'predictive entropy')
print("Predictive Entropy")
print("uce = ", np.round(uce, 2))


uce  = calibration(path, np.array(error_all), np.array(mi_all), n_bins, x_label = 'mutual information')
print("Mutual Information")
print("uce = ", np.round(uce, 2))


uce  = calibration(path, np.array(error_all), np.array(aleat_all), n_bins, x_label = 'average entropy')
print("Average Entropy")
print("uce = ", np.round(uce, 2))
