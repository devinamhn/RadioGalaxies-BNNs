import torch
import torchvision.transforms as T

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from cata2data import CataData
from torch.utils.data import DataLoader
from radiogalaxies_bnns.datasets.mightee import MighteeZoo
from radiogalaxies_bnns.inference.utils import Path_Handler

from radiogalaxies_bnns.inference.models import LeNet, LeNetDrop
import radiogalaxies_bnns.inference.nonBayesianCNN.utils as cnn_utils
from radiogalaxies_bnns.eval.uncertainty.uncertainty import entropy_MI, overlapping, GMM_logits, calibration

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

transform = T.Compose(
    [
        T.ToTensor(),
        T.Resize(150),  # Rescale to adjust for resolution difference between MIGHTEE & RGZ - was 70
        T.Normalize((1.59965605788234e-05,), (0.0038063037602458706,)),
    ]
)
paths = Path_Handler()._dict()
set = 'certain'

data = MighteeZoo(path=paths["mightee"], transform=transform, set="certain")
test_loader = DataLoader(data, batch_size=len(data))
for i, (x_test, y_test) in enumerate(test_loader):
    x_test, y_test = x_test.to(device), y_test.to(device)

print(len(data))
print(len(y_test))

model = LeNet(in_channels = 1, output_size =  2).to(device)
# path_out = config_dict['output']['path_out']
path_out = './radiogalaxies_bnns/results/nonBayesianCNNdatashuffle/cnn_'
criterion = torch.nn.CrossEntropyLoss()


n_ensembles = 10

softmax_ensemble, logits_ensemble = cnn_utils.eval_ensemble(model, test_loader, device, n_ensembles, path_out, criterion)
softmax_ensemble = torch.reshape(softmax_ensemble, (len(y_test), n_ensembles, 2))
logits_ensemble = torch.reshape(logits_ensemble, (len(y_test), n_ensembles, 2))


avg_error_mean = []

error_all = []
entropy_all = []
mi_all = []
aleat_all = []

fr1 = 0
fr2 = 0

loss_all = [] 

color = []
for k in range(len(y_test)):
    # print('galaxy', k)

    x = [0, 1]
    x = np.tile(x, (n_ensembles, 1))

    target = y_test[k].cpu().detach().numpy()

    if(target == 0):
        fr1+= 1
    elif(target==1):
        fr2+= 1
    
    # logits = pred_list[burn:][:, k:(k+1), :2].detach().numpy()
    # softmax_values = F.softmax(pred_list[burn:][:, k:(k+1), :2], dim =-1).detach().numpy()

    softmax_values = softmax_ensemble[k:k+1, :, :2].cpu().detach().numpy()
    loss = criterion(logits_ensemble[k:k+1, :, :2][0], torch.tile(y_test[k], (n_ensembles, 1)).flatten())

    sorted_softmax_values, lower_index, upper_index, mean_samples = credible_interval(softmax_values[:, :, 0].flatten(), 1) #0.64
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
    loss_all.append(loss.item())

print(fr1, fr2)
print("mean and std of error")
# print(error_all)
print("mean", np.round(np.mean(error_all)*100, 3))
print("std", np.round(np.std(error_all), 3 ))


print("Average CE Loss")
# print(loss_all)
print("mean", np.round(np.mean(loss_all), 3))
print("std", np.round(np.std(loss_all), 3 ))

fr1_start = 0 #{0}
# fr1_end = fr1 #{49, 68}
# fr2_start = fr1 #49, 68
# fr2_end = len(y_test) #len(val_indices) #{104, 145}
    
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

exit()

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    return img

for b, (im_batch, y_batch) in enumerate(DataLoader(data, batch_size=16)):
        fig = plt.figure(figsize=(13.0, 13.0))
        fig.subplots_adjust(0, 0, 1, 1)
        grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0)
        
        for i, (ax, im, y) in enumerate(zip(grid, list(im_batch), list(y_batch))):

            # im = im_batch[i].squeeze().numpy() 
            im = im.squeeze().numpy()
            
            ax.axis("off")
            text = f"FR{y+1}"

            ax.text(1, 66, text, fontsize=23, color="yellow")

            # contours
            threshold = 1
            ax.contour(np.where(im > threshold, 1, 0), cmap="cool", alpha=0.1)

            ax.imshow(im, cmap="hot")

        plt.axis("off")
        pil_img = fig2img(fig)
        plt.savefig(f"samples/{set}_{b}highres.png")
        plt.close(fig)