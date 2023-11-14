import torch
import torchvision
import torch.nn.functional as F
import torch.distributions as dists
import matplotlib.pyplot as plt

from galaxy_mnist import GalaxyMNIST, GalaxyMNISTHighrez
from torch.utils.data import DataLoader

import utils
from models import LeNetDrop

from netcal.metrics import ECE

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
        outputs = model(x_test)
        pred = F.softmax(outputs, dim = -1).argmax(dim=-1)
        error = (pred!= y_test).to(torch.float32).mean()
        running_error += error
    return running_error/(i+1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_dict, config = utils.parse_config('config_mb_dropout.txt')
seed = 123 #config['training']['seed']
torch.manual_seed(seed)

transform = torchvision.transforms.Compose([ torchvision.transforms.ToTensor(),
torchvision.transforms.Resize((150,150)), 
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

images, labels = test_dataset[0]
print(images.shape, images.dtype)

labels_map = {
    0: "smooth_round",
    1: "smooth_cigar",
    2: "edge_on_disk",
    3: "unbarred_spiral",
 }

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3

print(len(images))

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(2000, size=(1,)).item()
    print(sample_idx)
    #img, label = training_data[sample_idx]
    # transformed_image = transform(test_image)
    # image = images[sample_idx][None, :, :, :]
    # print(image.shape, image.dtype)
    # img = transform(images[sample_idx]).mT
    # label = labels[sample_idx].item()
    img, label = test_dataset[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze())
plt.savefig('./galaxy_mnist.png')

exit()
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
targets = torch.cat([y for x, y in test_loader], dim=0).to(device)#.numpy()


model = LeNetDrop(in_channels = 1, output_size =  2, dropout_rate = 0.5 ).to(device) #MLP(150, 200, 10)
model.load_state_dict(torch.load('./dropout/exp3/model', map_location=torch.device('cpu') ))


test_error = eval(test_loader)

print('Test error: {} %'.format(test_error*100))

# test_err = torch.zeros(20)
# for i in range(20):
#     test_err[i] = eval(test_loader)

# err_mean = torch.mean(test_err)
# err_std = torch.std(test_err)
# print('Test error mean: {} % '.format(err_mean*100))
# print('Test error std: {} % '.format(err_std))
@torch.no_grad()
def predict(test_loader, model):

    model.train(False)
    enable_dropout(model)
    probs_map =  []
    preds = []
    for i, (x_test, y_test) in enumerate(test_loader):
        x_test, y_test = x_test.to(device), y_test.to(device)
        outputs = model(x_test)
        prob = F.softmax(outputs, dim = -1)
        pred = prob.argmax(dim=-1)
        # error = (pred!= y_test).to(torch.float32).mean()
        probs_map.append(prob)
        preds.append(pred)
    return torch.cat(probs_map), torch.cat(preds)


probs_map, preds = predict(test_loader, model)

print((preds == targets)*1)
acc_map = ((preds == targets)*1).float().mean()
ece_map = ECE(bins=8).measure(probs_map.cpu().numpy(), targets.cpu().numpy())

targets = torch.ones((2000))
nll_map = -dists.Categorical(probs_map).log_prob(targets).mean()
print(f'[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}')




exit()
