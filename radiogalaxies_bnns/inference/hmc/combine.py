import torch
# torch.multiprocessing.set_sharing_strategy("file_system")
import os
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
from radiogalaxies_bnns.inference.utils import Path_Handler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

paths = Path_Handler()._dict()
path = paths['project'] / 'results' / 'inits/'  #'./results/leapfrog/' #intis/'#/thin #thin10000/'
n_chains = 5 #2 - intis #10 #10
num_samples = 200000
checkpt = 1000
thin1 = 1000 # thin every 1000 samples
thin2 = 500
n_files = 200 #190 #200 #int(num_samples/checkpt)

#generate folder list
folder_list = [f"{path}mb_out_{i}" for i in range(1, 6)]  #range(1, n_chains+1)]
print(folder_list)

#generate file list for each folder
file_list = []
for folder in folder_list:
    i=0
    for i in range(n_files): #(10, 50) to remove burn-in
        filenames = f"{folder}/params_hmc_{i}.pt" #[f"{folder}/params_hmc_{i}.pt" for i in range(1, n_files)]
        # print(filenames)
        file_list.append(filenames)

print(file_list)
#for just one chain
# file_list = file_list[int(n_files*0) : int(n_files*1)] #- (0, 1), (1, 2), (2,3) #[path + 'mb_out_1/params_hmc_0.pt', path + 'mb_out_1/params_hmc_1.pt']
# print(file_list)

for j in range(n_chains):

    file_list_ = file_list[int(n_files*j) : int(n_files*(j+1))] 
    i=0
    for file in file_list_:
        var_name = "params_hmc_{}".format(i)

        var_name1 = "params_hmc1_{}".format(i)
        var_name2 = "params_hmc2_{}".format(i)

        locals()[var_name] = torch.load(file, map_location=torch.device(device))

        locals()[var_name1] = locals()[var_name][0:1000:thin1]
        # locals()[var_name2] = locals()[var_name][0:1000:thin2]

        if(i==0):
            params_hmc_thin1 = locals()[var_name1]
            # params_hmc_thin2 = locals()[var_name2]
        else:
            params_hmc_thin1+= locals()[var_name1] 
            # params_hmc_thin2+= locals()[var_name2]
        print(i)

        del locals()[var_name]
        del locals()[var_name1]
        # del locals()[var_name2]
        i+=1


    torch.save(params_hmc_thin1, path+'/thin_chain'+str(j)) 

    # torch.save(params_hmc_thin1, path+'/thin500/thin_chain'+str(j)) 
    # torch.save(params_hmc_thin2, path+'/thin1000/thin_chain'+str(j))   
