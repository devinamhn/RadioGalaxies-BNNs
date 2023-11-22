import torch
import torch.optim as optim
import torch.nn.functional as F

def enable_dropout(m):
  for each_module in m.modules():
    if each_module.__class__.__name__.startswith('Dropout'):
      each_module.train()


def train(model, optimizer, criterion, train_loader, device):

    running_loss = 0.0
    for i, (x_train, y_train) in enumerate(train_loader):
        x_train, y_train = x_train.to(device), y_train.to(device)
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss/(i+1)

def validate(model, criterion, validation_loader, device):
    running_loss = 0.0
    running_error = 0.0
    enable_dropout(model)
    with torch.no_grad():
        for i, (x_val, y_val) in enumerate(validation_loader):
            x_val, y_val = x_val.to(device), y_val.to(device)
            outputs = model(x_val)
            loss = criterion(outputs, y_val)
            pred = F.softmax(outputs, dim = -1).argmax(dim=-1)
            error = (pred!= y_val).to(torch.float32).mean()
            running_loss += loss.item()
            running_error += error
    return running_loss/(i+1), running_error/(i+1)

def eval(model, test_loader, device):

    model.train(False)
    running_error = 0.0
    enable_dropout(model)
    with torch.no_grad():
        for i, (x_test, y_test) in enumerate(test_loader):
            x_test, y_test = x_test.to(device), y_test.to(device)
            outputs = model(x_test)
            pred = F.softmax(outputs, dim = -1).argmax(dim=-1)
            error = (pred!= y_test).to(torch.float32).mean()
            running_error += error

    return running_error/(i+1)
