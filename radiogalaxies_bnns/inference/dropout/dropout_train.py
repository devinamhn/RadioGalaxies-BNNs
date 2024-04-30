import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb

import radiogalaxies_bnns.inference.utils as utils
import radiogalaxies_bnns.inference.dropout.dropout_utils as dropout_utils

from radiogalaxies_bnns.inference.models import LeNetDrop
from radiogalaxies_bnns.inference.datamodules import MNISTDataModule, MiraBestDataModule, testloader_mb_uncert

def enable_dropout(m):
  for each_module in m.modules():
    if each_module.__class__.__name__.startswith('Dropout'):
      each_module.train()

jobid = int(sys.argv[1])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_dict, config = utils.parse_config('/share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/inference/dropout/config_mb_dropout.txt')
seed = config_dict['training']['seed'] + jobid #config['training']['seed']
seed_dataset = config_dict['training']['seed_dataset'] + jobid
torch.manual_seed(seed + jobid)
path_out = config_dict['output']['path_out']

lr = config_dict['training']['learning_rate']
weight_decay = config_dict['training']['weight_decay']
factor = config_dict['training']['factor']
patience = config_dict['training']['patience']
dropout_rate = config_dict['training']['dropout_rate']
epochs = config_dict['training']['epochs']

wandb_name = 'Dropout2 shuffle'+ str(jobid)
wandb.init(
    project= "Evaluating-BNNs",
    config = {
        "seed": seed,
        "data_seed": seed_dataset,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "factor": factor,
        "patience": patience,
        "epochs": epochs,
        "dropout": dropout_rate,
    },
    name=wandb_name,
)

datamodule = MiraBestDataModule(config_dict, hmc=False, seed_dataset=seed_dataset)
train_loader = datamodule.train_dataloader()
validation_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()

model = LeNetDrop(in_channels = 1, output_size =  2, dropout_rate = dropout_rate ).to(device) #MLP(150, 200, 10)
model_path = path_out+ str(jobid)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay) #1e-6 optim.SGD(model.parameters(), lr = 1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = factor, patience = patience)

best_vloss = 1_000_000.

train_loss = np.zeros(epochs)
val_loss = np.zeros(epochs)
val_error = np.zeros(epochs)


for epoch in range(epochs):

    model.train(True)
    avg_loss_train = dropout_utils.train(model, optimizer, criterion, train_loader, device)

    model.train(False)
    avg_loss_val, avg_error_val = dropout_utils.validate(model, criterion, validation_loader, device)

    # print('Epoch {}: LOSS train {} valid {}'.format(epoch, avg_loss_train, avg_loss_val))


    scheduler.step(avg_loss_val)
    # Track best performance, and save the model's state
    if avg_loss_val < best_vloss:
        best_vloss = avg_loss_val
        best_epoch = epoch
        torch.save(model.state_dict(), model_path+'/model')
    
    train_loss[epoch] = avg_loss_train
    val_loss[epoch] = avg_loss_val
    val_error[epoch] = avg_error_val*100
    
    wandb.log({"train_loss":avg_loss_train, "validation_loss":avg_loss_val, "validation_error":avg_error_val*100})

print('Finished Training')
wandb.log({"best_vloss_epoch": best_epoch, "best_vloss": best_vloss})


torch.save(train_loss, model_path + '/train_loss.pt')
torch.save(val_loss, model_path + '/val_loss.pt')
torch.save(val_error, model_path + '/val_error.pt')

model.load_state_dict(torch.load(model_path+'/model'))
test_error = dropout_utils.eval(model, test_loader, device)

print('Test error: {} %'.format(test_error*100))

test_err = torch.zeros(200)
for i in range(200):
    test_err[i] = dropout_utils.eval(model, test_loader, device)

err_mean = torch.mean(test_err)
err_std = torch.std(test_err)
print('Test error mean: {} % '.format(err_mean*100))
print('Test error std: {} % '.format(err_std))

wandb.log({"Mean test error":err_mean*100, "Std test error": err_std})
wandb.finish()