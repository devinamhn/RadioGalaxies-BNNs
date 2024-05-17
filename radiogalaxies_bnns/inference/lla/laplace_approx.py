""" Calculate last layer Laplace approximation for 10 experimental runs 
of non-Bayesian CNN
"""

import numpy as np
import torch
import torch.distributions as dists
from radiogalaxies_bnns.inference.models import LeNet
import radiogalaxies_bnns.inference.utils as utils
from radiogalaxies_bnns.inference.datamodules import MiraBestDataModule, testloader_mb_uncert
from radiogalaxies_bnns.eval.uncertainty.uncertainty import entropy_MI, calibration

from laplace import Laplace
from netcal.metrics import ECE

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

    return sorted_samples, index_lower, index_upper, mean_samples

paths = utils.Path_Handler()._dict()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_dict, config = utils.parse_config(paths['inference'] /'nonBayesianCNN'/'config_cnn.txt')
seed = 122 #config['training']['seed']
torch.manual_seed(seed)

datamodule = MiraBestDataModule(config_dict, hmc=False)
train_loader = datamodule.train_dataloader()
validation_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()
path_out = config_dict['output']['path_out']

print('TEST DATASET')
print(config_dict['output']['test_data'])

targets = torch.cat([y for x, y in test_loader], dim=0).to(device)#.numpy()


@torch.no_grad()
def predict(dataloader, model, laplace=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x.cuda()))
        else:
            py.append(torch.softmax(model(x.cuda()), dim=-1))

    return torch.cat(py)#.cpu().numpy()

# posterior_precision = la.posterior_precision
# posterior_std = torch.sqrt(1/posterior_precision)
# print(la.sample(200)[:, 0:168].shape)

# samples  = la.sample(200)[:, 0:168]
# torch.save(samples, './results/laplace/lla_samples.pt')

test_loader, test_data1, data_type, test_data = testloader_mb_uncert(config_dict['output']['test_data'], config_dict['data']['datadir'])
mean_expected_error_arr, std_expected_error_arr, uce_pe_arr, uce_mi_arr, uce_ae_arr = np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)


for n_ensemble in np.arange(0, 10, 1):
    model = LeNet(in_channels = 1, output_size =  2).to(device)
    model_path = path_out+ str(n_ensemble+1) +'/model'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu') ))

    # Laplace
    la = Laplace(model, 'classification',
                subset_of_weights= 'last_layer',
                hessian_structure='diag') # {diag, kron, full}
    la.fit(train_loader)
    la.optimize_prior_precision(pred_type = 'nn', method='marglik', verbose = True)

    probs_laplace = predict(test_loader, la, laplace=True)
    acc_laplace = ((probs_laplace.argmax(-1) == targets)*1).float().mean()
    ece_laplace = ECE(bins=8).measure(probs_laplace.cpu().numpy(), targets.cpu().numpy())
    nll_laplace = -dists.Categorical(probs_laplace).log_prob(targets).mean()

    print(f'[Diag Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}')


    num_samples = 200
    error_all = []
    entropy_all = []
    mi_all = []
    aleat_all =[]
    avg_error_mean = []

    fr1 = 0
    fr2 = 0

    indices = np.arange(0, len(test_data), 1)

    #for each sample in the test set:
    for index in indices:


        x_test = torch.unsqueeze(torch.tensor(test_data[index][0]),0)
        y_test = torch.unsqueeze(torch.tensor(test_data[index][1]),0)


        target = y_test.detach().numpy().flatten()[0]

        if(target == 0):
            fr1+= 1
        elif(target==1):
            fr2+= 1

        softmax_values = la.predictive_samples(x_test.cuda(), pred_type = 'nn', n_samples = num_samples )
        logits = la.predictive_samples(x_test.cuda(), pred_type = 'nn', n_samples = num_samples, return_logits = True )
        softmax_ = softmax_values.cpu()
        logits_ = logits.cpu()

    
        pred = softmax_.argmax(dim=-1).numpy().flatten()
    

        y_test_all = np.tile(y_test.detach().numpy().flatten()[0], num_samples)

        
        softmax = np.array(softmax_.detach().numpy())
        y_logits = np.array(logits_.detach().numpy())
        
        sorted_softmax_values, lower_index, upper_index, mean_samples = credible_interval(softmax[ :, :, 0].flatten(), 0.64)
        
        sorted_softmax_values_fr1 = sorted_softmax_values[lower_index:upper_index]
        sorted_softmax_values_fr2 = 1 - sorted_softmax_values[lower_index:upper_index]

    
        softmax_mean = np.vstack((mean_samples, 1-mean_samples)).T
        softmax_credible = np.vstack((sorted_softmax_values_fr1, sorted_softmax_values_fr2)).T   


        
        entropy, mutual_info, entropy_singlepass = entropy_MI(softmax_credible, 
                                                        samples_iter= len(softmax_credible[:,0]))

        pred = np.argmax(softmax_credible, axis = 1)

        y_test_all = np.tile(y_test, len(softmax_credible[:,0]))

        errors =  np.mean((pred != y_test_all).astype('uint8'))

    
        pred_mean = np.argmax(softmax_mean, axis = 1)
        error_mean = (pred_mean != target)*1

        error_all.append(errors)
        avg_error_mean.append(error_mean)
        entropy_all.append(entropy/np.log(2))    
        mi_all.append(mutual_info/np.log(2))
        aleat_all.append(entropy_singlepass/np.log(2))



    path = './'

    fr1_start = 0 #{0}
    fr1_end = fr1 #{49, 68}
    fr2_start = fr1 #49, 68
    fr2_end = len(indices) #len(val_indices) #{104, 145}
        
    n_bins = 8
        
    uce_pe  = calibration(path, np.array(error_all), 
                        np.array(entropy_all), n_bins, x_label = 'predictive entropy')
    uce_mi  = calibration(path, np.array(error_all), 
                        np.array(mi_all), n_bins, x_label = 'mutual information')
    uce_ae  = calibration(path, np.array(error_all), 
                        np.array(aleat_all), n_bins, x_label = 'average entropy')
        
    mean_expected_error = np.array(avg_error_mean).mean()*100 
    std_expected_error = np.array(avg_error_mean).std()

    mean_expected_error_arr[n_ensemble], std_expected_error_arr[n_ensemble], uce_pe_arr[n_ensemble], uce_mi_arr[n_ensemble], uce_ae_arr[n_ensemble] = mean_expected_error, std_expected_error, uce_pe, uce_mi, uce_ae

print("Mean of error over seeds", mean_expected_error_arr.mean())
print("Std of error over seeds", mean_expected_error_arr.std())

print("Mean of std over seeds", std_expected_error_arr.mean())
print("Std of std over seeds", std_expected_error_arr.std())


print('Mean of uce pe', uce_pe_arr.mean())
print('Std of uce pe', uce_pe_arr.std())

print('Mean of uce mi', uce_mi_arr.mean())
print('Std of uce mi', uce_mi_arr.std())

print('Mean of uce ae', uce_ae_arr.mean())
print('Std of uce ae', uce_ae_arr.std())

exit()




