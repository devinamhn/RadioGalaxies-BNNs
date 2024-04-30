import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv

from radiogalaxies_bnns.eval.uncertainty.uncertainty import entropy_MI, overlapping, GMM_logits

from radiogalaxies_bnns.inference.datamodules import MNISTDataModule, MiraBestDataModule, testloader_mb_uncert
from radiogalaxies_bnns.inference.models import LeNetDrop
import radiogalaxies_bnns.inference.utils as utils
import radiogalaxies_bnns.inference.dropout.dropout_utils as dropout_utils

def energy_function(logits, T = 1):
    
    # print(-torch.logsumexp(pred_list_mbconf, dim = 2))
    mean_energy = torch.mean(-torch.logsumexp(logits, dim = 2), dim = 0).detach().numpy()
    std_energy = torch.std(-torch.logsumexp(logits, dim = 2), dim = 0).detach().numpy()

    return mean_energy


def credible_interval(samples, credibility):
    '''
    calculate credible interval - equi-tailed interval instead of highest density interval

    samples values and indices
    '''
    mean_samples = samples.mean()
    sorted_samples = np.sort(samples)
    lower_bound = 0.5 * (1 - credibility)
    upper_bound = 0.5 * (1 + credibility)

    index_lower = int(np.round(len(samples) * lower_bound))
    index_upper = int(np.round(len(samples) * upper_bound))

    # print('lower', index_lower, 'upper', index_upper)

    return sorted_samples, index_lower, index_upper, mean_samples


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_dict, config = utils.parse_config('/share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/inference/dropout/config_mb_dropout.txt')
seed = 122 #config['training']['seed']
torch.manual_seed(seed)
path_out = config_dict['output']['path_out']
dropout_rate = config_dict['training']['dropout_rate']


print('TEST DATASET')
print(config_dict['output']['test_data'])
test_loader, test_data1, data_type, test_data = testloader_mb_uncert(config_dict['output']['test_data'], config_dict['data']['datadir'])

for i, (x_test, y_test) in enumerate(test_loader):
    x_test, y_test = x_test.to(device), y_test.to(device)
   
# datamodule = MiraBestDataModule(config_dict, hmc=False)
# train_loader = datamodule.train_dataloader()
# validation_loader = datamodule.val_dataloader()
# test_loader = datamodule.test_dataloader()
    
# model_path = path_out + str(8)
# model = LeNetDrop(in_channels = 1, output_size =  2, dropout_rate = dropout_rate ).to(device) #MLP(150, 200, 10)
# model.load_state_dict(torch.load(model_path+'/model', map_location=torch.device('cpu') ))

# criterion = torch.nn.CrossEntropyLoss()

# model.load_state_dict(torch.load('./dropout/mlp/model', map_location=torch.device('cpu') ))
# print(model.out.weight.flatten().detach().numpy())
# map_weights = model.out.weight.flatten().detach().numpy()
# np.save('./dropout/map_samples_l7_weights', map_weights)

#to save model weights from last layer
# model.load_state_dict(torch.load('./dropout/exp3/model', map_location=torch.device('cpu') ))
# enable_dropout(model)
# dropout_weights = model.out.weight.flatten().detach().numpy()
# np.save('./dropout/dropout_samples_l7_weights', dropout_weights)


# exit()

test_dataloader, test_data1, data_type, test_data= testloader_mb_uncert(config_dict['output']['test_data'], config_dict['data']['datadir'])

# utils.calibration_test(path_out, model, test_data_uncert = 'MBFRConfident', device=device, path=config_dict['data']['datadir'], test = True)

mean_expected_error, std_expected_error, uce_pe, uce_mi, uce_ae = np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)


# uce_pe_arr = []

for i in range(10):
    model_path = path_out + str(i+1)
    model = LeNetDrop(in_channels = 1, output_size =  2, dropout_rate = dropout_rate ).to(device) #MLP(150, 200, 10)
    model.load_state_dict(torch.load(model_path+'/model', map_location=torch.device(device) ))

    mean_expected_error[i], std_expected_error[i], uce_pe[i], uce_mi[i], uce_ae[i] = utils.calibration_test(path_out, model, test_data_uncert = 'MBFRConfident', device=device, path=config_dict['data']['datadir'], test = True)

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

###########################
exit()

###uncertanity tests###
model.load_state_dict(torch.load('./dropout/exp2/model'))

test_error = eval(test_loader)

csvfile = config_dict['output']['filename_uncert']   
path_out = config_dict['output']['path_out']
cvsfile = path_out + csvfile

test_dataloader, test_data1, data_type, test_data= testloader_mb_uncert(config_dict['output']['test_data'], config_dict['data']['datadir'])


rows = ['index', 'target', 'entropy', 'entropy_singlepass' , 'mutual info', 'var_logits_0','var_logits_1','softmax_eta', 'logits_eta','cov0_00','cov0_01','cov0_11','cov1_00','cov1_01','cov1_11', 'data type', 'label']
                                    
with open(csvfile, 'w+', newline="") as f_out:
    writer = csv.writer(f_out, delimiter=',')
    writer.writerow(rows)

indices = np.arange(0, len(test_data), 1)
# print(indices)
for index in (indices):
    
    x = torch.unsqueeze(torch.tensor(test_data[index][0]),0)
    y = torch.unsqueeze(torch.tensor(test_data[index][1]),0)
    #print("target is",y)
    target = y.detach().numpy().flatten()[0]
    samples_iter = 50
    #for a single datapoint
    with torch.no_grad():
        output_ = []
        logits_=[]
        prediction=[]
        y_test_all = []
        
        model.train(False)
        enable_dropout(model)

        for j in range(samples_iter):
            x_test, y_test = x.to(device), y.to(device)

            outputs = model(x_test)
            softmax = F.softmax(outputs, dim = -1)
            pred = softmax.argmax(dim=-1)


            output_.append(softmax.cpu().detach().numpy().flatten())
            logits_.append(outputs.cpu().detach().numpy().flatten())
            
            # predict = pred.mean(dim=0).argmax(dim=-1)

            prediction.append(pred.cpu().detach().numpy().flatten()[0])
            y_test_all.append(y_test.cpu().detach().numpy().flatten()[0])

    print(output_)
    softmax = np.array(output_)#.cpu().detach().numpy())
    y_logits = np.array(logits_)#.cpu().detach().numpy())
        
    mean_logits = np.mean(y_logits,axis=0)
    var_logits = np.std(y_logits,axis=0)

    softmax_list =  softmax
    logits_list = y_logits

    # print(logits_list)
    # print(softmax_list)
   
    x = [0, 1]
    # y = softmax[0]
    x = np.tile(x, (samples_iter, 1))
    # print(y)
    # print(y_test)

    target = y_test.cpu().detach().numpy()

    if(target == 0):
        if(data_type == 'Hybrid'):
            label = 'Conf'
        else:
            label = 'FRI'
        
    elif(target ==1):
        if(data_type == 'Hybrid'):
            label = 'Uncert'
        else:
            label = 'FRII'
    

    plt.figure(figsize= (2.6, 4.8), dpi=300)
    plt.rcParams["axes.grid"] = False
    plt.subplot((211))
    plt.scatter(x, softmax_list, marker='_',linewidth=1,color='b',alpha=0.5)
    plt.title("softmax outputs")
    plt.xticks(np.arange(0, 2, 1.0))

        
    plt.subplot((212))
    plt.imshow(test_data1[index][0])
    plt.axis("off")
    #label = 'target = ' + str(test_data1[index][1])
    plt.title('class'+str(target) +': '+label)
    # plt.show()
    plt.savefig(path_out+ 'softmax_test' + str(index)+ '.png')

    softmax = np.array(softmax_list)
    logits = np.array(logits_list) 
    # print(np.array(softmax_list))
    # print(softmax.shape)
    # print(softmax)
    # # print(softmax_list[:,])
    # print(softmax[:,0])

    mean_logits = np.mean(logits,axis=0)
    var_logits = np.std(logits,axis=0)
    print("Mean of Logits", mean_logits)
    print("Stdev pf Logits", var_logits)

    entropy, mutual_info, entropy_singlepass = entropy_MI(softmax, samples_iter)

    print("Entropy:", entropy)
    print("Mutual Information:", mutual_info)
    print("Entropy of a single pass:", entropy_singlepass)

    softmax_eta = overlapping(softmax[:,0], softmax[:,1])
    print("Softmax Overlap Index", softmax_eta)
        
    logits_eta = overlapping(logits[:,0], logits[:,1])
    print("Logit-Space Overlap Index", logits_eta)
        
    covs = GMM_logits(logits, 2)

    # plt.figure(dpi=300)
    # covs = GMM_logits(logits, 2)
    # plt.savefig(path_out+'cov' + str(index)+ '.png')

    
    plt.figure(dpi=200)
    plt.rcParams["axes.grid"] = False
    plt.axes().set_facecolor('white')
    plt.scatter(x, logits, marker='_',linewidth=1,color='b',alpha=0.5)
    plt.xticks(np.arange(0, 2, 1))
    plt.savefig(path_out+'logits' + str(index)+ '.png')

    # data_type = 'MBConf'
    # ['index', 'target', 'entropy','entropy_singlepass ', 'mutual info', 'var_logits_0','var_logits_1','softmax_eta', 'logits_eta','GMM_covs', 'data type', 'label']
    _results = [index, target, entropy, entropy_singlepass, mutual_info, var_logits[0],var_logits[1], softmax_eta, logits_eta,covs[0][0][0], covs[0][0][1], covs[0][1][1],covs[1][0][0],covs[1][0][1],covs[1][1][1], data_type, label ]
        
    with open(csvfile, 'a', newline="") as f_out:
        writer = csv.writer(f_out, delimiter=',')
        writer.writerow(_results)
