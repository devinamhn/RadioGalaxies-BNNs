import argparse
import configparser as ConfigParser 
import ast
import torch
import numpy as np 
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler
import torch.nn.functional as F
import torchvision
from pathlib import Path
import torch.utils.data as D

from radiogalaxies_bnns.datasets import mirabest
from radiogalaxies_bnns.eval.uncertainty.uncertainty import entropy_MI, calibration
from galaxy_mnist import GalaxyMNIST, GalaxyMNISTHighrez

from cata2data import CataData
from torch.utils.data import DataLoader
from radiogalaxies_bnns.datasets.mightee import MighteeZoo
# from radiogalaxies_bnns.inference.utils import Path_Handler



def parse_args():
    """
        Parse the command line arguments
        """
    parser = argparse.ArgumentParser()
    parser.add_argument('-C','--config', default="config_bbb.txt", required=True, help='Name of the input config file')
    
    args, __ = parser.parse_known_args()
    
    return vars(args)

# -----------------------------------------------------------

def parse_config(filename):
    
    config = ConfigParser.ConfigParser(allow_no_value=True)
    config.read(filename)
    
    # Build a nested dictionary with tasknames at the top level
    # and parameter values one level down.
    taskvals = dict()
    for section in config.sections():
        
        if section not in taskvals:
            taskvals[section] = dict()
        
        for option in config.options(section):
            # Evaluate to the right type()
            try:
                taskvals[section][option] = ast.literal_eval(config.get(section, option))
            except (ValueError,SyntaxError):
                err = "Cannot format field '{0}' in config file '{1}'".format(option,filename)
                err += ", which is currently set to {0}. Ensure strings are in 'quotes'.".format(config.get(section, option))
                raise ValueError(err)

    return taskvals, config

class Path_Handler:
    """Handle and generate paths in project directory"""

    def __init__(
        self, **kwargs
    ):  # use defaults except where specified in kwargs e.g. Path_Handler(data=some_alternative_dir)
        path_dict = {}
        path_dict["root"] = kwargs.get("root", Path(__file__).resolve().parent.parent.parent)
        path_dict["project"] = kwargs.get(
            "project", Path(__file__).resolve().parent.parent
        )  # i.e. this repo

        path_dict["data"] = kwargs.get("data", path_dict["root"] / "radiogalaxies_bnns" / "data")

        for key, path_str in path_dict.copy().items():
            path_dict[key] = Path(path_str)

        self.path_dict = path_dict

    def fill_dict(self):
        """Create dictionary of required paths"""

        self.path_dict["rgz"] = self.path_dict["data"] / "rgz"
        self.path_dict["mb"] = self.path_dict["data"] / "mb"
        self.path_dict["mightee"] = self.path_dict["data"] / "MIGHTEE"

    def create_paths(self):
        """Create missing directories"""
        for path in self.path_dict.values():
            create_path(path)

    def _dict(self):
        """Generate path dictionary, create any missing directories and return dictionary"""
        self.fill_dict()
        self.create_paths()
        return self.path_dict


def create_path(path):
    if not Path.exists(path):
        Path.mkdir(path)



def enable_dropout(m):
  for each_module in m.modules():
    if each_module.__class__.__name__.startswith('Dropout'):
      each_module.train()

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


def calibration_test(path_out, model, test_data_uncert, device, path, test):

    test_data = test_data_uncert
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.0031 ,), (0.0350,))])
    test_data1 = test_data_uncert
    
    # test = "combined" #{True, False, "combined"}
    # for test_data and test_data1
    
    if(test_data_uncert == 'MBFRConfident'):
        
        if(test == True):
        # confident test set
            test_data = mirabest.MBFRConfident(path, train=False,
                             transform=transform, target_transform=None,
                             download=False)
            
            test_data1 = mirabest.MBFRConfident(path, train=False,
                             transform=None, target_transform=None,
                             download=False)
            #uncomment for test set
            indices = np.arange(0, len(test_data), 1)
  
    elif(test_data_uncert == 'MBFRUncertain'):
        # uncertain
        
        test_data = mirabest.MBFRUncertain(path, train=False,
                         transform=transform, target_transform=None,
                         download=False)
        
        test_data1 = mirabest.MBFRUncertain(path, train=False,
                         transform=None, target_transform=None,
                         download=False)
        data_type = 'MBFR_Uncert'
    elif(test_data_uncert == 'MBHybrid'):
        #hybrid
        test_data = mirabest.MBHybrid(path, train=True,
                         transform=transform, target_transform=None,
                         download=False)
        test_data1 = mirabest.MBHybrid(path, train=True,
                         transform=None, target_transform=None,
                         download=False)
        data_type = 'MBHybrid'

    # elif(test_data_uncert == 'Galaxy_MNIST'):


    else:
        print("Test data for uncertainty quantification misspecified")
    indices = np.arange(0, len(test_data), 1)
    logit = True
    
  
    num_batches_test = 1
    
    error_all = []
    entropy_all = []
    mi_all = []
    aleat_all =[]
    avg_error_mean = []
    loss_all = []
    
    fr1 = 0
    fr2 = 0
    
    for index in indices:
        
        x = torch.unsqueeze(torch.tensor(test_data[index][0]),0)
        y = torch.unsqueeze(torch.tensor(test_data[index][1]),0)
        target = y.detach().numpy().flatten()[0]
        samples_iter = 200
        
        if(target == 0):
            fr1+= 1
        elif(target==1):
            fr2+= 1
        
        output_ = []
        logits_=[]
        y_test_all = []
        prediction = []
        errors = []

        #for a single datapoint
        with torch.no_grad():
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

        softmax = np.array(output_)#.cpu().detach().numpy())
        y_logits = np.array(logits_)#.cpu().detach().numpy())

        sorted_softmax_values, lower_index, upper_index, mean_samples = credible_interval(softmax[:, 0].flatten(), 0.64)
        # print("90% credible interval for FRI class softmax", sorted_softmax_values[lower_index], sorted_softmax_values[upper_index])
        
        sorted_softmax_values_fr1 = sorted_softmax_values[lower_index:upper_index]
        sorted_softmax_values_fr2 = 1 - sorted_softmax_values[lower_index:upper_index]

        softmax_mean = np.vstack((mean_samples, 1-mean_samples)).T
        softmax_credible = np.vstack((sorted_softmax_values_fr1, sorted_softmax_values_fr2)).T
        
        # entropy, mutual_info, entropy_singlepass = entropy_MI(softmax, samples_iter)
        entropy, mutual_info, entropy_singlepass = entropy_MI(softmax_credible, 
                                                        samples_iter= len(softmax_credible[:,0]))

        pred = np.argmax(softmax_credible, axis = 1)
        # print(y_test, pred)

        y_test_all = np.tile(y_test, len(softmax_credible[:,0]))

        errors =  np.mean((pred != y_test_all).astype('uint8'))

        pred_mean = np.argmax(softmax_mean, axis = 1)
        # print(pred_mean, np.argmax(softmax_mean))
        error_mean = (pred_mean != target)*1
        
        error_all.append(errors)
        avg_error_mean.append(error_mean)
        entropy_all.append(entropy/np.log(2))    
        mi_all.append(mutual_info/np.log(2))
        aleat_all.append(entropy_singlepass/np.log(2))
        # loss_all.append(loss.item())
    
    fr1_start = 0 #{0}
    fr1_end = fr1 #{49, 68}
    fr2_start = fr1 #49, 68
    fr2_end = len(indices) #len(val_indices) #{104, 145}
    
    # print(fr1_start, fr1_end, fr2_start, fr2_end)
  
    n_bins = 8
    
    uce_pe  = calibration(path_out, np.array(error_all), np.array(entropy_all), n_bins, x_label = 'predictive entropy')
    # print("Predictive Entropy")
    # print("uce = ", np.round(uce_pe, 2))

    uce_mi  = calibration(path_out, np.array(error_all), np.array(mi_all), n_bins, x_label = 'mutual information')
    # print("Mutual Information")
    # print("uce = ", np.round(uce_mi, 2))

    
    uce_ae  = calibration(path_out, np.array(error_all), np.array(aleat_all), n_bins, x_label = 'average entropy')
    # print("Average Entropy")
    # print("uce = ", np.round(uce_ae, 2))

    print("mean and std of error")
    print(error_all)
    print(np.mean(error_all)*100)
    print(np.std(error_all))
    mean_err_all = np.mean(error_all)*100
    sts_err_all = np.std(error_all)

    print("Average of expected error")
    print((np.array(avg_error_mean)).mean()*100)
    print((np.array(avg_error_mean)).std())

    mean_expected_error = np.array(avg_error_mean).mean()*100 
    std_expected_error = np.array(avg_error_mean).std()

    return mean_expected_error, std_expected_error, uce_pe, uce_mi, uce_ae

def get_test_data(test_data_uncert, path, device):
    # test_data = test_data_uncert
    if(test_data_uncert == 'MBFRConfident'):
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.0031 ,), (0.0350,))])
        test_data = mirabest.MBFRConfident(path, train=False,
                            transform=transform, target_transform=None,
                            download=False)
    elif(test_data_uncert == 'MBFRUncertain'):
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.0031 ,), (0.0350,))])
        test_data = mirabest.MBFRUncertain(path, train=False,
                         transform=transform, target_transform=None,
                         download=False)
    elif(test_data_uncert == 'MBHybrid'):
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.0031 ,), (0.0350,))])
        test_data = mirabest.MBHybrid(path, train=True,
                         transform=transform, target_transform=None,
                         download=False)
    elif(test_data_uncert == 'Galaxy_MNIST'):
        transform = torchvision.transforms.Compose([ torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((150,150), antialias = True), 
        torchvision.transforms.Grayscale(),
        ])
        train_dataset = GalaxyMNISTHighrez(
            root='/share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/data/dataGalaxyMNISTHighres',
            download=True,
            train=True,  # by default, or set False for test set
            transform = transform
        )
        test_dataset = GalaxyMNISTHighrez(
            root='/share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/data/dataGalaxyMNISTHighres',
            download=True,
            train=False,  # by default, or set False for test set
            transform = transform
        )
        gal_mnist_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=104, shuffle = False)

        for i, (x_test_galmnist, y_test_galmnist) in enumerate(gal_mnist_test_loader):
            x_test_galmnist, y_test_galmnist = x_test_galmnist.to(device), y_test_galmnist.to(device)
            y_test_galmnist = torch.zeros(104).to(device)
            if(i==0):
                break
        test_data = x_test_galmnist

    elif(test_data_uncert == 'mightee'):
        transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(150),  # Rescale to adjust for resolution difference between MIGHTEE & RGZ - was 70
            torchvision.transforms.Normalize((1.59965605788234e-05,), (0.0038063037602458706,)),
        ]
        )
        paths = Path_Handler()._dict()
        set = 'certain'
        data = MighteeZoo(path=paths["mightee"], transform=transform, set="certain")
        test_loader = DataLoader(data, batch_size=len(data))     
        test_data = data

    else:
        print("Test data for uncertainty quantification misspecified")    

    return test_data

def get_logits(model, test_data_uncert, device, path):
    test_data = get_test_data(test_data_uncert, path, device)

    indices = np.arange(0, len(test_data), 1)
    logit = True
  
    num_batches_test = 1
    
    
    fr1 = 0
    fr2 = 0
    samples_iter = 200

    output_ = torch.zeros(samples_iter, len(test_data), 2)
    
    logits_all= []
    for index in indices:
        
        x = torch.unsqueeze(test_data[index][0].clone().detach(), 0) #torch.unsqueeze(torch.tensor(test_data[index][0]),0)
        if(test_data_uncert == 'Galaxy_MNIST'):
            y = torch.tensor(0)#test_data[index][1]),0)
        else:
            y = torch.unsqueeze(torch.tensor(test_data[index][1]),0)
        target = y.detach().numpy().flatten()[0]
        logits_=[]
        #for a single datapoint
        with torch.no_grad():
     
            model.train(False)
            enable_dropout(model)

            for j in range(samples_iter):
                x_test, y_test = x.to(device), y.to(device)

                outputs = model(x_test)
                softmax = F.softmax(outputs, dim = -1)
                pred = softmax.argmax(dim=-1)

                output_[j][index] = outputs

    # print(output_)
    return output_


def get_logits_mlp(model, test_data_uncert, device, path):
    test_data = get_test_data(test_data_uncert, path, device)

    indices = np.arange(0, len(test_data), 1)
    logit = True
    num_batches_test = 1
    
    fr1 = 0
    fr2 = 0
    output_ = torch.zeros(len(test_data), 2)
    
    # print(indices)
    logits_all= []
    for index in indices:
        
        x = torch.unsqueeze(test_data[index][0].clone().detach(), 0) #torch.unsqueeze(torch.tensor(test_data[index][0]),0)
        if(test_data_uncert == 'Galaxy_MNIST'):
            y = torch.tensor(0)#test_data[index][1]),0)
        else:
            y = torch.unsqueeze(torch.tensor(test_data[index][1]),0)
        target = y.detach().numpy().flatten()[0]
        logits_=[]
        #for a single datapoint
        with torch.no_grad():
     
            model.train(False)
            # enable_dropout(model)

            # for j in range(samples_iter):
            x_test, y_test = x.to(device), y.to(device)

            outputs = model(x_test)
            # softmax = F.softmax(outputs, dim = -1)
            # pred = softmax.argmax(dim=-1)

            output_[index] = outputs

    # print(output_.shape)
    return output_

   
def get_logits_ensembles(model, test_data_uncert, device, path, n_ensembles, path_out):
    test_data = get_test_data(test_data_uncert, path, device)
    indices = np.arange(0, len(test_data), 1)
    logit = True
  
    num_batches_test = 1
    
    
    fr1 = 0
    fr2 = 0
    # samples_iter = 200
    samples_iter = n_ensembles
    output_ = torch.zeros(samples_iter, len(test_data), 2)
    
    # print(indices)
    logits_all= []
    for index in indices:
        
        x = torch.unsqueeze(test_data[index][0].clone().detach(), 0) #torch.unsqueeze(torch.tensor(test_data[index][0]),0)
        if(test_data_uncert == 'Galaxy_MNIST'):
            y = torch.tensor(0)#test_data[index][1]),0)
        else:
            y = torch.unsqueeze(torch.tensor(test_data[index][1]),0)
        target = y.detach().numpy().flatten()[0]
        logits_=[]
        #for a single datapoint
        with torch.no_grad():
     
            model.train(False)
            # enable_dropout(model)

            for j in range(samples_iter):

                model_path = path_out+ str(j+1) +'/model'
                model.load_state_dict(torch.load(model_path))

                x_test, y_test = x.to(device), y.to(device)

                outputs = model(x_test)
                softmax = F.softmax(outputs, dim = -1)
                pred = softmax.argmax(dim=-1)

                output_[j][index] = outputs

    return output_

def get_logits_la(la, test_data_uncert, device, path):
    test_data = get_test_data(test_data_uncert, path, device)
    indices = np.arange(0, len(test_data), 1)
    logit = True
  
    num_batches_test = 1
        
    fr1 = 0
    fr2 = 0
    samples_iter = 200

    output_ = torch.zeros(samples_iter, len(test_data), 2)
    
    # print(indices)
    logits_all= []
    for index in indices:
        
        x = torch.unsqueeze(test_data[index][0].clone().detach(), 0) #torch.unsqueeze(torch.tensor(test_data[index][0]),0)
        if(test_data_uncert == 'Galaxy_MNIST'):
            y = torch.tensor(0)#test_data[index][1]),0)
        else:
            y = torch.unsqueeze(torch.tensor(test_data[index][1]),0)
        target = y.detach().numpy().flatten()[0]
        logits_=[]
        #for a single datapoint
        # with torch.no_grad():
            # model.train(False)
            # enable_dropout(model)
     
        # for i, (x_test, y_test) in enumerate(test_loader):
        #     x_test, y_test = x_test.to(device), y_test.to(device)
        #     print(x_test.shape)
        # print(x.shape)
        outputs = la.predictive_samples(x.cuda(), pred_type = 'nn', n_samples = 200, return_logits = True )
        # print(outputs.shape)
        output_[:, index :, ] = outputs
            # for j in range(samples_iter):
            #     x_test, y_test = x.to(device), y.to(device)

            #     outputs = model(x_test)
            #     softmax = F.softmax(outputs, dim = -1)
            #     pred = softmax.argmax(dim=-1)

            #     output_[j][index] = outputs

    # print(output_)
    return output_

def rgz_cut(rgz_dset, threshold, mb_cut: bool = True, remove_duplicates=False):
    """Cut rgz data-set based on angular size and whether data-point is contained in MiraBest"""

    n = len(rgz_dset)
    idx_bool = np.ones(n, dtype=bool)
    idx = np.arange(n)

    if remove_duplicates:
        idx_bool = np.zeros(n, dtype=bool)
        _, idx_unique = np.unique(rgz_dset.data, axis=0, return_index=True)
        idx_bool[idx_unique] = True

        print(f"Removed {n - np.count_nonzero(idx_bool)} duplicate samples")
        n = np.count_nonzero(idx_bool)

    idx_bool *= rgz_dset.sizes > threshold
    print(f"Removing {n - np.count_nonzero(idx_bool)} samples below angular size threshold.")
    n = np.count_nonzero(idx_bool)

    if mb_cut:
        idx_bool *= rgz_dset.mbflg == 0

        # Print number of MB samples removed
        print(f"Removed {n - np.count_nonzero(idx_bool)} MiraBest samples from RGZ")

    idx = np.argwhere(idx_bool)

    subset = D.Subset(rgz_dset, idx)
    print(f"RGZ dataset cut from {n} to {len(subset)} samples")
    return subset
