import argparse
import configparser as ConfigParser 
import ast
import torch
import numpy as np 
import torchvision.transforms as transforms
import mirabest
from uncertainty import entropy_MI
from uncertainty import calibration
from torch.utils.data import SubsetRandomSampler
import torch.nn.functional as F
import torchvision
from galaxy_mnist import GalaxyMNIST, GalaxyMNISTHighrez




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



def flatten(model):
    return torch.nn.utils.parameters_to_vector(model.parameters())

def unflatten(params, model):
    
    torch.nn.utils.vector_to_parameters(params, model.parameters()) 
    return None

# def flatten(model):
#     return torch.cat([p.flatten() for p in model.parameters()]).requires_grad_()


def unflatten_hamil(model, flattened_params):
    if flattened_params.dim() != 1:
        raise ValueError('Expecting a 1d flattened_params')
    params_list = []
    i = 0
    for val in list(model.parameters()):
        length = val.nelement()
        param = flattened_params[i:i+length].view_as(val)
        params_list.append(param)
        i += length
    
    #params_tensor = torch.tensor(params_list, dtype=torch.float32)

    return params_list#params_tensor

#################################################################################
# Found here: https://gist.github.com/apaszke/4c8ead6f17a781d589f6655692e7f6f0
#################################################################################

import sys
import types
from collections import OrderedDict

PY2 = sys.version_info[0] == 2
_internal_attrs = {'_backend', '_parameters', '_buffers', '_backward_hooks', '_forward_hooks', '_forward_pre_hooks', '_modules'}


### Had to add this for conv net
_new_methods = {'conv2d_forward','_forward_impl', '_check_input_dim', '_conv_forward',
                'check_forward_args', 'check_input', 'check_hidden_size', "get_expected_hidden_size",
                "get_expected_cell_size", "permute_hidden"}


class Scope(object):
    def __init__(self):
        self._modules = OrderedDict()


# Function keeps looping and turning each module in the network to a function
def _make_functional(module, params_box, params_offset):
    self = Scope()
    num_params = len(module._parameters)
    param_names = list(module._parameters.keys())
    # Set dummy variable to bias_None to rename as flag if no bias
    if 'bias' in param_names and module._parameters['bias'] is None:
        param_names[-1] = 'bias_None' # Remove last name (hopefully bias) from list
    forward = type(module).forward.__func__ if PY2 else type(module).forward
    if type(module) == torch.nn.modules.container.Sequential:
        # Patch sequential model by replacing the forward method
        forward = Sequential_forward_patch
    if 'BatchNorm' in module.__class__.__name__:
        # Patch sequential model by replacing the forward method (hoping applies for all BNs need
        # to put this in tests)
        forward = bn_forward_patch

    for name, attr in module.__dict__.items():
        if name in _internal_attrs:
            continue   #If internal attributes skip
        setattr(self, name, attr)
    ### Had to add this for conv net (MY ADDITION)
    for name in dir(module):
        if name in _new_methods:
            if name == '_conv_forward': # Patch for pytorch 1.5.0+cu101
                setattr(self, name, types.MethodType(type(module)._conv_forward,self))
            if name == 'conv2d_forward':
                setattr(self, name, types.MethodType(type(module).conv2d_forward,self))
            if name == '_forward_impl':
                setattr(self, name, types.MethodType(type(module)._forward_impl,self))
            if name == 'check_forward_args':
                setattr(self, name, types.MethodType(type(module).check_forward_args,self))
            if name == 'check_input':
                setattr(self, name, types.MethodType(type(module).check_input,self))
            if name == 'check_hidden_size':
                setattr(self, name, types.MethodType(type(module).check_hidden_size,self))
            if name == 'get_expected_hidden_size':
                setattr(self, name, types.MethodType(type(module).get_expected_hidden_size,self))
            if name == 'get_expected_cell_size':
                setattr(self, name, types.MethodType(type(module).get_expected_cell_size,self))
            if name == 'permute_hidden':
                setattr(self, name, types.MethodType(type(module).permute_hidden,self))
            if name == '_check_input_dim': # Batch Norm
                # import pdb; pdb.set_trace()
                setattr(self, name, types.MethodType(type(module)._check_input_dim,self))

    child_params_offset = params_offset + num_params
    for name, child in module.named_children():
        child_params_offset, fchild = _make_functional(child, params_box, child_params_offset)
        self._modules[name] = fchild  # fchild is functional child
        setattr(self, name, fchild)
    def fmodule(*args, **kwargs):

        # Uncomment below if statement to step through (with 'n') assignment of parameters.
#         if params_box[0] is not None:
#             import pdb; pdb.set_trace()

        # If current layer has no bias, insert the corresponding None into params_box
        # with the params_offset ensuring the correct weight is applied to the right place.
        if 'bias_None' in param_names:
            params_box[0].insert(params_offset + 1, None)
        for name, param in zip(param_names, params_box[0][params_offset:params_offset + num_params]):

            # In order to deal with layers that have no bias:
            if name == 'bias_None':
                setattr(self, 'bias', None)
            else:
                setattr(self, name, param)
        # In the forward pass we receive a context object and a Tensor containing the
        # input; we must return a Tensor containing the output, and we can use the
        # context object to cache objects for use in the backward pass.

        # When running the kwargs no longer exist as they were put into params_box and therefore forward is just
        # forward(self, x), so I could comment **kwargs out
        return forward(self, *args) #, **kwargs)

    return child_params_offset, fmodule


def make_functional(module):
    params_box = [None]
    _, fmodule_internal = _make_functional(module, params_box, 0)

    def fmodule(*args, **kwargs):
        params_box[0] = kwargs.pop('params') # if key is in the dictionary, remove it and return its value, else return default. If default is not given and key is not in the dictionary, a KeyError is raised.
        return fmodule_internal(*args, **kwargs)

    return fmodule

##### PATCH FOR nn.Sequential #####

def Sequential_forward_patch(self, input):
    # put at top of notebook nn.Sequential.forward = Sequential_forward_patch
    for label, module in self._modules.items():
        input = module(input)
    return input

##### Patch for batch norm #####
def bn_forward_patch(self, input):
    # set running var to None and running mean
    return torch.nn.functional.batch_norm(
                input, running_mean = None, running_var = None,
                weight = self.weight, bias = self.bias,
                training = self.training,
                momentum = self.momentum, eps = self.eps)

def gpu_check_delete(string, locals):
    if string in locals:
        del locals[string]
        torch.cuda.empty_cache()

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
    
    #     elif(test == False):
    #         #validation set
    #         train_data_confident = mirabest.MBFRConfident(path, train=True,
    #                                           transform=transform, target_transform=None,
    #                                           download=False)
    #         train_data_conf = train_data_confident
            
    #         #train valid test
    #         dataset_size = len(train_data_conf)
    #         indices = list(range(dataset_size))
    #         split = int(dataset_size*0.2) #int(np.floor(validation_split * dataset_size))
    #         shuffle_dataset = True
    #         random_seed = 15
    #         batch_size = 50
    #         if shuffle_dataset :
    #             np.random.seed(random_seed)
    #             np.random.shuffle(indices)
    #         train_indices, val_indices = indices[split:], indices[:split]
                
    #         # Creating data samplers and loaders:
    #         #train_sampler = SubsetRandomSampler(train_indices)
    #         valid_sampler = SubsetRandomSampler(val_indices)
                
    #         #train_loader = torch.utils.data.DataLoader(train_data_conf, batch_size=batch_size, sampler=train_sampler)
    #         #validation_loader = torch.utils.data.DataLoader(train_data_conf, batch_size=batch_size, sampler=valid_sampler)
            
    #         #print(val_indices)
    #         #validation_loader = train_data_confident[val_indices]
            
    #         test_data = torch.utils.data.Subset(train_data_conf, val_indices)

    #         #test_loader = torch.utils.data.DataLoader(dataset=test_data_conf, batch_size=batch_size,shuffle=True)

    #         #indices = np.sort(val_indices)
            
    #         fr1_all = 0
    #         fr2_all = 0
            
    #         #y_test_all_arr = []
    #         index_fr1, index_fr2 = [], []
    #         for index in np.arange(0, len(test_data), 1):
                
    #             y = torch.unsqueeze(torch.tensor(test_data[index][1]),0)
    #             target = y.detach().numpy().flatten()[0]
    #             #print(target)
    #             if(target == 0):
    #                 fr1_all+= 1
    #                 index_fr1.append(index)
                    
    #             elif(target==1):
    #                 fr2_all+= 1
    #                 index_fr2.append(index)
                    
    #         index_sorted = np.append(np.array(index_fr1), np.array(index_fr2))
    #         #print(index_sorted)
    #         #print("fr1_all=", fr1_all, "fr2_all=", fr2_all, "ratio =", fr1_all/fr2_all)
            
    #         for index in index_sorted:
    #             y = torch.unsqueeze(torch.tensor(test_data[index][1]),0)
    #             target = y.detach().numpy().flatten()[0]
    #             #print(target)
  
    #         indices = index_sorted
    #         #print(indices)
    #     elif(test == "combined"):
    #         test_data_ = mirabest.MBFRConfident(path, train=False,
    #                         transform=transform, target_transform=None,
    #                         download=False)
            
    #         #validation set
    #         train_data_confident = mirabest.MBFRConfident(path, train=True,
    #                                           transform=transform, target_transform=None,
    #                                           download=False)
    #         train_data_conf = train_data_confident
            
    #         #train valid test
    #         dataset_size = len(train_data_conf)
    #         indices = list(range(dataset_size))
    #         split = int(dataset_size*0.2) #int(np.floor(validation_split * dataset_size))
    #         shuffle_dataset = True
    #         random_seed = 15
    #         batch_size = 50
    #         if shuffle_dataset :
    #             np.random.seed(random_seed)
    #             np.random.shuffle(indices)
    #         train_indices, val_indices = indices[split:], indices[:split]
                
    #         # Creating data samplers and loaders:
    #         #train_sampler = SubsetRandomSampler(train_indices)
            
    #         #valid_sampler = SubsetRandomSampler(val_indices)
            
    #         #train_loader = torch.utils.data.DataLoader(train_data_conf, batch_size=batch_size, sampler=train_sampler)
    #         #validation_loader = torch.utils.data.DataLoader(train_data_conf, batch_size=batch_size, sampler=val_indices)
            
    #         val_indices = np.sort(val_indices)
    #         #print(val_indices)
            
    #         #validation_loader = train_data_confident[val_indices]
            
    #         #test_data = train_data_confident
    #         #test_loader = torch.utils.data.DataLoader(dataset=test_data_conf, batch_size=batch_size,shuffle=True)
    #         val_data =torch.utils.data.Subset(train_data_conf, val_indices)
            
    #         test_data = torch.utils.data.ConcatDataset([val_data, test_data_])
            
    #         #print(len(test_data))
            
    #         indices = np.arange(0, len(test_data), 1)
            
    #         fr1_all = 0
    #         fr2_all = 0
            
    #         #y_test_all_arr = []
    #         index_fr1, index_fr2 = [], []
    #         for index in np.arange(0, len(test_data), 1):
                
    #             y = torch.unsqueeze(torch.tensor(test_data[index][1]),0)
    #             target = y.detach().numpy().flatten()[0]
    #             #print(target)
    #             if(target == 0):
    #                 fr1_all+= 1
    #                 index_fr1.append(index)
                    
    #             elif(target==1):
    #                 fr2_all+= 1
    #                 index_fr2.append(index)
                    
    #         index_sorted = np.append(np.array(index_fr1), np.array(index_fr2))
    #         #print(index_sorted)
    #         #print("fr1_all=", fr1_all, "fr2_all=", fr2_all, "ratio =", fr1_all/fr2_all)
            
    #         for index in index_sorted:
    #             y = torch.unsqueeze(torch.tensor(test_data[index][1]),0)
    #             target = y.detach().numpy().flatten()[0]
    #             #print(target)
                
            
    #         #print(indices)
    #         indices = index_sorted
                    
    #     data_type = 'MBFR_Conf'
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
    
    fr1 = 0
    fr2 = 0
    
    print(indices)

    for index in indices:
        
        x = torch.unsqueeze(torch.tensor(test_data[index][0]),0)
        y = torch.unsqueeze(torch.tensor(test_data[index][1]),0)
        #print("target is",y)
        target = y.detach().numpy().flatten()[0]
        samples_iter = 200
        #print(target)
        
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
            #accs_ =[]
            # i = 1
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


        # softmax=[]
        # logit_arr = []
        # for i in range(samples_iter):
        #     a = np.exp(np.array(output_[i][0]))
        #     softmax.append(a[0])
        #     logit_arr.append(np.array(logits_[i][0]))

        # print("CHECKING LOGITS AND OUTPUT")
        # print(index)
        # print(logits_)
        # print(output_)
        # print(prediction)        

        softmax = np.array(output_)#.cpu().detach().numpy())
        y_logits = np.array(logits_)#.cpu().detach().numpy())
        
        print(softmax.shape)

        sorted_softmax_values, lower_index, upper_index, mean_samples = credible_interval(softmax[:, 0].flatten(), 0.64)
        print("90% credible interval for FRI class softmax", sorted_softmax_values[lower_index], sorted_softmax_values[upper_index])
        
        sorted_softmax_values_fr1 = sorted_softmax_values[lower_index:upper_index]
        sorted_softmax_values_fr2 = 1 - sorted_softmax_values[lower_index:upper_index]

        # print(sorted_softmax_values_fr1)
        # print(sorted_softmax_values_fr2)
        softmax_mean = np.vstack((mean_samples, 1-mean_samples)).T
        softmax_credible = np.vstack((sorted_softmax_values_fr1, sorted_softmax_values_fr2)).T   
   
        # mean_logits = np.mean(y_logits,axis=0)
        # var_logits = np.std(y_logits,axis=0)
        #print("Mean of Logits", mean_logits)
        #print("Stdev pf Logits", var_logits)
        
        # entropy, mutual_info, entropy_singlepass = entropy_MI(softmax, samples_iter)
        entropy, mutual_info, entropy_singlepass = entropy_MI(softmax_credible, 
                                                        samples_iter= len(softmax_credible[:,0]))

        pred = np.argmax(softmax_credible, axis = 1)
        print(y_test, pred)

        y_test_all = np.tile(y_test, len(softmax_credible[:,0]))

        errors =  np.mean((pred != y_test_all).astype('uint8'))

        print(softmax_mean)
        print(target)
        pred_mean = np.argmax(softmax_mean, axis = 1)
        print(pred_mean, np.argmax(softmax_mean))
        error_mean = (pred_mean != target)*1

        print(error_mean)
    
        #print("Entropy:", entropy)
        #print("Mutual Information:", mutual_info)
        #print("Entropy of a single pass:", entropy_singlepass)
        
        
        # prediction = np.array(prediction)
        # y_test_all = np.array(y_test_all)
        
        
        # errors = np.mean((prediction != y_test_all).astype('uint8'))
        print(errors)
        
        error_all.append(errors)
        avg_error_mean.append(error_mean)
        
        entropy_all.append(entropy/np.log(2))    
        mi_all.append(mutual_info/np.log(2))
        aleat_all.append(entropy_singlepass/np.log(2))
    
    #print("Validation: fr1=", fr1, " fr2=", fr2, " ratio", fr1/fr2)
    #print("Train: fr1=", fr1_all - fr1, " fr2=",fr2_all- fr2, " ratio",(fr1_all - fr1)/(fr2_all- fr2) )
    
    #print(len(error_all[0:49]), len(error_all[49:104]))    
    #print(error_all)
    #print(entropy_all)
    '''
    outputs_ = np.array(output_)
    prediction = np.array(prediction)
    softmax= np.exp(outputs_)
    y_test_all = np.array(y_test_all)
    logits = np.array(logits)
    #print(output_)
    #print(softmax)
    #print(y_test_all)
    #print(prediction)
    '''
    
    
    
    fr1_start = 0 #{0}
    fr1_end = fr1 #{49, 68}
    fr2_start = fr1 #49, 68
    fr2_end = len(indices) #len(val_indices) #{104, 145}
    
    print(fr1_start, fr1_end, fr2_start, fr2_end)
    
    print(error_all)
    #entropy, mutual_info, entropy_singlepass = entropy_MI(softmax, samples_iter)
    n_bins = 8
    
    uce  = calibration(path_out, np.array(error_all), np.array(entropy_all), n_bins, x_label = 'predictive entropy')
    print("Predictive Entropy")
    print("uce = ", np.round(uce, 2))
    uce_0  = calibration(path_out, np.array(error_all[fr1_start:fr1_end]), np.array(entropy_all[fr1_start:fr1_end]), n_bins, x_label = 'predictive entropy') 
    print("UCE FRI= ", np.round(uce_0, 2))
    uce_1  = calibration(path_out, np.array(error_all[fr2_start:fr2_end]), np.array(entropy_all[fr2_start:fr2_end]), n_bins, x_label = 'predictive entropy')
    print("UCE FRII = ", np.round(uce_1, 2))
    cUCE = (uce_0 + uce_1)/2 
    print("cUCE=", np.round(cUCE, 2))
    
    #max_mi = np.amax(np.array(mi_all))
    #print("max mi",max_mi)
    #mi_all = mi_all/max_mi
    #print(mi_all)
    #print("max mi",np.amax(mi_all))
    uce  = calibration(path_out, np.array(error_all), np.array(mi_all), n_bins, x_label = 'mutual information')
    print("Mutual Information")
    print("uce = ", np.round(uce, 2))
    uce_0  = calibration(path_out, np.array(error_all[fr1_start:fr1_end]), np.array(mi_all[fr1_start:fr1_end]), n_bins, x_label = 'mutual information') 
    print("UCE FRI= ", np.round(uce_0, 2))
    uce_1  = calibration(path_out, np.array(error_all[fr2_start:fr2_end]), np.array(mi_all[fr2_start:fr2_end]), n_bins, x_label = 'mutual information')
    print("UCE FRII = ", np.round(uce_1, 2))
    cUCE = (uce_0 + uce_1)/2 
    print("cUCE=", np.round(cUCE,2))  
    
    
    uce  = calibration(path_out, np.array(error_all), np.array(aleat_all), n_bins, x_label = 'average entropy')
    print("Average Entropy")
    print("uce = ", np.round(uce, 2))
    uce_0  = calibration(path_out, np.array(error_all[fr1_start:fr1_end]), np.array(aleat_all[fr1_start:fr1_end]), n_bins, x_label = 'average entropy') 
    print("UCE FRI= ",np.round(uce_0, 2))
    uce_1  = calibration(path_out, np.array(error_all[fr2_start:fr2_end]), np.array(aleat_all[fr2_start:fr2_end]), n_bins, x_label = 'average entropy')
    print("UCE FRII = ", np.round(uce_1, 2))
    cUCE = (uce_0 + uce_1)/2 
    print("cUCE=", np.round(cUCE, 2))  
    
    print("mean and std of error")
    print(error_all)
    print(np.mean(error_all)*100)
    print(np.std(error_all))

    print("Average of expected error")
    print((np.array(avg_error_mean)).mean())
    print((np.array(avg_error_mean)).std())


def get_logits(model, test_data_uncert, device, path):

    test_data = test_data_uncert
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.0031 ,), (0.0350,))])
    test_data1 = test_data_uncert
    
    if(test_data_uncert == 'MBFRConfident'):
        
    #confident test set
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

    elif(test_data_uncert == 'Galaxy_MNIST'):
        transform = torchvision.transforms.Compose([ torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((150,150), antialias = True), 
        torchvision.transforms.Grayscale(),
        ])
        # 64 pixel images
        train_dataset = GalaxyMNISTHighrez(
            root='./dataGalaxyMNISTHighres',
            download=True,
            train=True,  # by default, or set False for test set
            transform = transform
        )

        test_dataset = GalaxyMNISTHighrez(
            root='./dataGalaxyMNISTHighres',
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


    else:
        print("Test data for uncertainty quantification misspecified")
    indices = np.arange(0, len(test_data), 1)
    logit = True
  
    num_batches_test = 1
    
    
    fr1 = 0
    fr2 = 0
    samples_iter = 200

    output_ = torch.zeros(samples_iter, len(test_data), 2)
    
    print(indices)
    logits_all= []
    for index in indices:
        
        x = torch.unsqueeze(torch.tensor(test_data[index][0]),0)
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

    print(output_)
    return output_


def get_logits_mlp(model, test_data_uncert, device, path):

    test_data = test_data_uncert
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.0031 ,), (0.0350,))])
    test_data1 = test_data_uncert
    
    if(test_data_uncert == 'MBFRConfident'):
        
    #confident test set
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

    elif(test_data_uncert == 'Galaxy_MNIST'):
        transform = torchvision.transforms.Compose([ torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((150,150), antialias = True), 
        torchvision.transforms.Grayscale(),
        ])
        # 64 pixel images
        train_dataset = GalaxyMNISTHighrez(
            root='./dataGalaxyMNISTHighres',
            download=True,
            train=True,  # by default, or set False for test set
            transform = transform
        )

        test_dataset = GalaxyMNISTHighrez(
            root='./dataGalaxyMNISTHighres',
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


    else:
        print("Test data for uncertainty quantification misspecified")
    indices = np.arange(0, len(test_data), 1)
    logit = True
  
    num_batches_test = 1
    
    
    fr1 = 0
    fr2 = 0
    # samples_iter = 200

    output_ = torch.zeros(len(test_data), 2)
    
    # print(indices)
    logits_all= []
    for index in indices:
        
        x = torch.unsqueeze(torch.tensor(test_data[index][0]),0)
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

    print(output_.shape)
    return output_


        


        
