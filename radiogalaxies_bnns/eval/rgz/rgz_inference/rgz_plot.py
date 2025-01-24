from radiogalaxies_bnns.eval.rgz.rgz_inference.rgz_datamodules import RGZ_DataModule
from radiogalaxies_bnns.inference.utils import Path_Handler
from radiogalaxies_bnns.datasets.rgz108k import RGZ108k 
from torch.utils.data import DataLoader
import torchvision.transforms as T
from matplotlib import pyplot as plt 
import torch
from radiogalaxies_bnns.inference.models import LeNet
import hamiltorch
import pandas as pd
import numpy as np
import seaborn as sns

def energy_function(logits, T = 1):
    
    # print(-torch.logsumexp(pred_list_mbconf, dim = 2))
    mean_energy = torch.mean(-torch.logsumexp(logits, dim = 2), dim = 0).cpu().detach().numpy()
    std_energy = torch.std(-torch.logsumexp(logits, dim = 2), dim = 0).cpu().detach().numpy()

    return mean_energy


# Get paths
paths = Path_Handler()._dict()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Get energy scores
energy_score_df = pd.read_csv('radiogalaxies_bnns/results/ood/hmc_energy_scores_rgz.csv', index_col=0)

# Sort the DataFrame by 'HMC RGZ' energy values'
energy_score_df = energy_score_df.sort_values(by = 'HMC RGZ')

# Define the bin edges with intervals of 2.0
bin_edges = range(int(energy_score_df['HMC RGZ'].min()) - 2, int(energy_score_df['HMC RGZ'].max()) + 2, 2)

# Bin the values
energy_score_df['binned'] = pd.cut(energy_score_df['HMC RGZ'], bins=bin_edges)

# Pick 10 indices from each bin
sampled_indices = []
for bin in energy_score_df['binned'].unique():
    bin_indices = energy_score_df[energy_score_df['binned'] == bin].index
    sampled_indices.extend(bin_indices[:10])
    print(bin)

# Print the sampled indices and corresponding values
print("Sampled Indices and Corresponding Values:")
for idx in sampled_indices:
    print(f"Index: {idx}, Value: {energy_score_df.loc[idx, 'HMC RGZ']}")


# Define transform
transform = T.Compose(
    [
        # T.CenterCrop(70),
        T.ToTensor(),
        T.Normalize((0.008008896,), (0.05303395,)),
    ]
)
# Load in RGZ dataset
test_dataset = RGZ108k(paths["rgz"], train=True, transform=transform)

figure = plt.figure(figsize=(8*4, 6*3), dpi=200)
cols, rows = 6, 8

for i in range(0, cols * rows):
    sample_idx = sampled_indices[i] #torch.randint(200, size=(1,)).item()
    print(sample_idx)
    img, label = test_dataset[sample_idx]
    figure.add_subplot(rows, cols, i+1)
    plt.title(f"Idx: {sample_idx}, E: {round(energy_score_df.loc[sample_idx, 'HMC RGZ'],3)}")
    plt.axis("off")
    plt.imshow(img.permute(1, 2, 0))
    plt.savefig('./rgz_binned.png') 
    