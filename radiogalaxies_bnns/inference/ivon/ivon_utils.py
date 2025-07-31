import torch
import torch.optim as optim
import torch.nn.functional as F



def train(model, optimizer, criterion, train_loader, device, train_samples = 1):
    model.train(True)

    running_loss = 0.0
    for i, (x_train, y_train) in enumerate(train_loader):
        x_train, y_train = x_train.to(device), y_train.to(device)
        for _ in range(train_samples):
            with optimizer.sampled_params(train=True):

                
                outputs = model(x_train)
                loss = criterion(outputs, y_train)
                loss.backward()
            
        optimizer.step()
        optimizer.zero_grad()
        
        running_loss += loss.item()
    return running_loss/(i+1)

def validate(model, optimizer, criterion, validation_loader, device, train_samples = 1):
    running_loss = 0.0
    running_error = 0.0
    model.train(False)
    # enable_dropout(model)
    with torch.no_grad():
        for i, (x_val, y_val) in enumerate(validation_loader):
            x_val, y_val = x_val.to(device), y_val.to(device)
            for _ in range(train_samples):
                with optimizer.sampled_params(train=False):
                    outputs = model(x_val)
                    loss = criterion(outputs, y_val)
            pred = F.softmax(outputs, dim = -1).argmax(dim=-1)
            error = (pred!= y_val).to(torch.float32).mean()
            running_loss += loss.item()
            running_error += error

    return running_loss/(i+1), running_error/(i+1)


def eval(model, optimizer, test_loader, device, train_samples = 1):

    model.train(False)
    running_error = 0.0
    with torch.no_grad():
        for i, (x_test, y_test) in enumerate(test_loader):
            x_test, y_test = x_test.to(device), y_test.to(device)
            for _ in range(train_samples):
                with optimizer.sampled_params(train=False):
                    outputs = model(x_test)
            
            pred = F.softmax(outputs, dim = -1).argmax(dim=-1)
            error = (pred!= y_test).to(torch.float32).mean()
            running_error += error

    return running_error/(i+1)

# def eval(dataloader, model, loss_fn, test_mc):
#     model.eval()
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     test_loss, correct = 0, 0
#     sampled_probs_all = []
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             # Predict at mean
#             if test_mc == 0:
#                 logit = model(X)
#                 test_loss += loss_fn(logit, y).item()
#                 correct += (logit.argmax(1) == y).type(torch.float).sum().item()
#             # Predict with samples
#             else:
#                 sampled_probs = []
#                 for i in range(test_mc):
#                     with optimizer.sampled_params():
#                         sampled_logit = model(X)
#                         sampled_probs.append(F.softmax(sampled_logit, dim=1))

#                 sampled_probs_all.append(torch.stack(sampled_probs, dim = 0))
#                 prob = torch.mean(torch.stack(sampled_probs), dim=0)
#                 _, pred = prob.max(1)
#                 test_loss -= torch.sum(torch.log(prob.clamp(min=1e-6)) * F.one_hot(y, 2), dim=1).mean() #was one_hot(y, 10) for 10 classes in mnist
#                 correct += pred.eq(y).sum().item()

#     test_loss /= num_batches
#     correct /= size
#     if test_mc != 0:
        
#         print(f"\nIVON -- Test Performance with {test_mc:0d} Test Samples \n Accuracy: {(100 * correct):>0.2f}%, Avg loss: {test_loss:>7f} \n")
#     else:
#         print(f"IVON -- Test Performance with Mean Prediction \n Accuracy: {(100 * correct):>0.2f}%, Avg loss: {test_loss:>7f} \n")
#     if test_mc!=0:
#       return sampled_probs_all

