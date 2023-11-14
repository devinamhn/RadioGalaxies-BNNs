import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import torch.nn.utils.prune as prune
import os
import csv
import pandas as pd

import mirabest
import utils
from pathlib import Path

from matplotlib import pyplot as plt
import mirabest
from uncertainty import entropy_MI, overlapping, GMM_logits
import csv

config_dict, config = utils.parse_config('config1.txt')

#output
# filename = config_dict['output']['filename_uncert']
# test_data_uncert = config_dict['output']['test_data']

path = "C:/Users/devin/Documents/PhD/bnn_mcmc/bnn_mcmc/results/dropout/exp2/" #/uncert_mbconf/"
csvfile = 'mirabest_uncert_combined.csv'#'model_uncert_fr_combined.csv'   #"model_uncert_all_models.csv" #"/uncert_mbhybrid1000/mbhybrid_uncert.csv"  #"model_uncert_all_combined.csv" #"mirabest_uncert_all1000.csv"
data = pd.read_csv(path+csvfile)

# csv_vi = "model_uncert.csv"
# data_vi = pd.read_csv(path + csv_vi)


# plt.hist(data["entropy"], color= data["data type"])

x = "label" #{"pruning", "label","target", "data type", "type"}
hue = "model" #{"label", "data type", "type", "target", "model"}

figure = plt.figure(figsize = (6,4), dpi= 200)
ax = sns.boxplot( x = x,y="entropy",data=data, hue = hue,  palette="Set3")
# ax = sns.boxplot( x = x,y="entropy",data=data_vi, hue = hue,  palette="Set3")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(path+'entropy.png', bbox_inches = "tight")

figure = plt.figure(figsize = (6,4), dpi= 200)
ax = sns.boxplot( x = x,y="var_logits_0",data=data, hue = hue, palette="Set3")
pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(path+'var_logits_fr1.png', bbox_inches = "tight")

figure = plt.figure(figsize = (6,4), dpi= 200)
ax = sns.boxplot( x = x,y="var_logits_1",data=data, hue = hue, palette="Set3")
pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(path+'var_logits_fr2.png', bbox_inches = "tight")


# figure = plt.figure(dpi= 200)
# ax = sns.boxplot( x = x,y="entropy_singlepass",data=data, hue = hue, palette="Set3")
# pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# pl.savefig(path+'entropy_singlepass.png')

figure = plt.figure(figsize = (6,4), dpi= 200)
ax = sns.boxplot( x = x,y="mutual info",data=data, hue = hue, palette="Set3")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(path+'mutual_info.png', bbox_inches = "tight")

# figure = plt.figure(dpi= 200)
# ax = sns.boxplot( x = x,y="softmax_eta",data=data, hue=hue,palette="Set3")
# pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# figure = plt.figure(dpi= 200)
# ax = sns.boxplot( x = x,y="logits_eta",data=data,  palette="Set3")
# pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# figure = plt.figure(dpi= 200)
# ax = sns.boxplot( x = x,y="cov0_00",data=data, hue = hue, palette="Set3")
# pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# figure = plt.figure(dpi= 200)
# ax = sns.boxplot( x = x,y="cov0_01",data=data, hue = hue, palette="Set3")
# pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# figure = plt.figure(dpi= 200)
# ax = sns.boxplot( x = x,y="cov0_11",data=data, hue = hue, palette="Set3")
# pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# figure = plt.figure(dpi= 200)
# ax = sns.boxplot( x = x,y="cov1_00",data=data, hue = hue, palette="Set3")
# pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# figure = plt.figure(dpi= 200)
# ax = sns.boxplot( x = x,y="cov1_01",data=data, hue = hue, palette="Set3")
# pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# figure = plt.figure(dpi= 200)
# ax = sns.boxplot( x = x,y="cov1_11",data=data, hue = hue, palette="Set3")
# pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# #ax = sns.swarmplot(x="target", y="logits_eta", data=data, color=".25", size = 3)
