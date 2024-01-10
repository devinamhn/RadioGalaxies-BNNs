import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


import radiogalaxies_bnns.inference.utils as utils
import radiogalaxies_bnns.inference.ensembles.ensembles_utils as ensemble_utils

from radiogalaxies_bnns.inference.models import LeNet
from radiogalaxies_bnns.inference.datamodules import MNISTDataModule, MiraBestDataModule, testloader_mb_uncert


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_dict, config = utils.parse_config('/share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/inference/ensembles/config_ensembles.txt')
jobid = int(sys.argv[1])
seed = 123 + jobid #config['training']['seed']
torch.manual_seed(seed)
path_out = config_dict['output']['path_out']


datamodule = MiraBestDataModule(config_dict, hmc=False)
train_loader = datamodule.train_dataloader()
validation_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()

model = LeNet(in_channels = 1, output_size =  2).to(device) #MLP(150, 200, 10)
model_path = path_out+'model'+str(jobid)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay = 1e-5) #1e-6 optim.SGD(model.parameters(), lr = 1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.9, patience = 2)

epochs = config_dict['training']['epochs']

best_vloss = 1_000_000.

train_loss = np.zeros(epochs)
val_loss = np.zeros(epochs)
val_error = np.zeros(epochs)


for epoch in range(epochs):

    model.train(True)
    avg_loss_train = ensemble_utils.train(model, optimizer, criterion, train_loader, device)

    model.train(False)
    avg_loss_val, avg_error_val = ensemble_utils.validate(model, criterion, validation_loader, device)

    print('Epoch {}: LOSS train {} valid {}'.format(epoch, avg_loss_train, avg_loss_val))


    scheduler.step(avg_loss_val)
    # Track best performance, and save the model's state
    if avg_loss_val < best_vloss:
        best_vloss = avg_loss_val
        torch.save(model.state_dict(), model_path)
    
    train_loss[epoch] = avg_loss_train
    val_loss[epoch] = avg_loss_val
    val_error[epoch] = avg_error_val*100

print('Finished Training')

torch.save(train_loss, model_path + 'train_loss.pt')
torch.save(val_loss, model_path + 'val_loss.pt')
torch.save(val_error, model_path + 'val_error.pt')

plt.figure(dpi=200)
plt.plot(train_loss, label='train loss')
plt.plot(val_loss, label='val loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(model_path + 'loss.png')

plt.figure(dpi=200)
plt.plot(val_error, label='val error')
plt.legend(loc='upper right')
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.savefig(model_path + 'val_error.png')

model.load_state_dict(torch.load('/share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/results/ensembles/model' + str(jobid)))
test_error = ensemble_utils.eval(model, test_loader, device)

print('Test error: {} %'.format(test_error*100))

# test_err = torch.zeros(200)
# for i in range(200):
#     test_err[i] = dropout_utils.eval(model, test_loader, device)

# err_mean = torch.mean(test_err)
# err_std = torch.std(test_err)
# print('Test error mean: {} % '.format(err_mean*100))
# print('Test error std: {} % '.format(err_std))

