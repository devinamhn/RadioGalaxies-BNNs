import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from radiogalaxies_bnns.inference.utils import credible_interval, get_test_data, calibration
from radiogalaxies_bnns.eval.uncertainty.uncertainty import entropy_MI, overlapping, GMM_logits 

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


# def eval(model, optimizer, test_loader, device, train_samples = 1):

#     model.train(False)
#     running_error = 0.0
#     with torch.no_grad():
#         for i, (x_test, y_test) in enumerate(test_loader):
#             x_test, y_test = x_test.to(device), y_test.to(device)
#             for _ in range(train_samples):
#                 with optimizer.sampled_params(train=False):
#                     outputs = model(x_test)
            
#             pred = F.softmax(outputs, dim = -1).argmax(dim=-1)
#             error = (pred!= y_test).to(torch.float32).mean()
#             running_error += error

#     return running_error/(i+1)

def eval(dataloader, model, loss_fn, test_mc):

    model.train(False)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    sampled_probs_all = []
    error_all = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            # Predict at mean
            if test_mc == 0:
                logit = model(X)
                test_loss += loss_fn(logit, y).item()
                correct += (logit.argmax(1) == y).type(torch.float).sum().item()
            # Predict with samples
            else:
                sampled_probs = []
                for i in range(test_mc):
                    with optimizer.sampled_params(train=False):
                        sampled_logit = model(X)
                        sampled_probs.append(F.softmax(sampled_logit, dim=1))

                sampled_probs_all.append(torch.stack(sampled_probs, dim = 0))
                prob = torch.mean(torch.stack(sampled_probs), dim=0)
                _, pred = prob.max(1)
                test_loss -= torch.sum(torch.log(prob.clamp(min=1e-6)) * F.one_hot(y, 2), dim=1).mean() #was one_hot(y, 10) for 10 classes in mnist
                correct += pred.eq(y).sum().item()
                errors = (pred != y).to(torch.float32)
                error_all.append(errors)
    test_loss /= num_batches
    correct /= size
    
    if test_mc != 0:
        sampled_probs_all = torch.cat(sampled_probs_batches, dim=1)
        error_all = torch.cat(error_all, dim=0)
    
        print(f"\nIVON -- Test Performance with {test_mc:0d} Test Samples \n Accuracy: {(100 * correct):>0.2f}%, Avg loss: {test_loss:>7f} \n")
        return sampled_probs_all, error_all

    else:
        print(f"IVON -- Test Performance with Mean Prediction \n Accuracy: {(100 * correct):>0.2f}%, Avg loss: {test_loss:>7f} \n")


def calibration_test(path_out, model, optimizer, test_data_uncert, device, path, test, samples_iter = 200):
    #called in dropout_eval()
    test_data = get_test_data(test_data_uncert, path, device)

    indices = np.arange(0, len(test_data), 1)
    logit = True
  
    num_batches_test = 1
    
    error_all = []
    entropy_all = []
    mi_all = []
    aleat_all =[]
    avg_error_mean = []
    loss_all = []
    
    fr1 = 0
    fr2 = 0
    
    for index in indices:
        
        x = torch.unsqueeze(torch.tensor(test_data[index][0]),0)
        y = torch.unsqueeze(torch.tensor(test_data[index][1]),0)
        target = y.detach().numpy().flatten()[0]
        # samples_iter = 200
        
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
            model.train(False)
            # enable_dropout(model)

            for j in range(samples_iter):
                with optimizer.sampled_params(train=False):
                    x_test, y_test = x.to(device), y.to(device)

                    outputs = model(x_test)
                    softmax = F.softmax(outputs, dim = -1)
                    pred = softmax.argmax(dim=-1)


                    output_.append(softmax.cpu().detach().numpy().flatten())
                    logits_.append(outputs.cpu().detach().numpy().flatten())
                    
                    # predict = pred.mean(dim=0).argmax(dim=-1)

                    prediction.append(pred.cpu().detach().numpy().flatten()[0])
                    y_test_all.append(y_test.cpu().detach().numpy().flatten()[0])      

        softmax = np.array(output_)#.cpu().detach().numpy())
        y_logits = np.array(logits_)#.cpu().detach().numpy())

        sorted_softmax_values, lower_index, upper_index, mean_samples = credible_interval(softmax[:, 0].flatten(), 0.64)
        # print("90% credible interval for FRI class softmax", sorted_softmax_values[lower_index], sorted_softmax_values[upper_index])
        
        sorted_softmax_values_fr1 = sorted_softmax_values[lower_index:upper_index]
        sorted_softmax_values_fr2 = 1 - sorted_softmax_values[lower_index:upper_index]

        softmax_mean = np.vstack((mean_samples, 1-mean_samples)).T
        softmax_credible = np.vstack((sorted_softmax_values_fr1, sorted_softmax_values_fr2)).T
        
        # entropy, mutual_info, entropy_singlepass = entropy_MI(softmax, samples_iter)
        entropy, mutual_info, entropy_singlepass = entropy_MI(softmax_credible, 
                                                        samples_iter= len(softmax_credible[:,0]))

        pred = np.argmax(softmax_credible, axis = 1)
        # print(y_test, pred)
        
        #not sure is this is needed??
        y_test_all = np.tile(y_test.cpu().detach().numpy(), len(softmax_credible[:,0]))

        errors =  np.mean((pred != y_test_all).astype('uint8'))

        pred_mean = np.argmax(softmax_mean, axis = 1)
        # print(pred_mean, np.argmax(softmax_mean))
        error_mean = (pred_mean != target)*1
        
        error_all.append(errors)
        avg_error_mean.append(error_mean)
        entropy_all.append(entropy/np.log(2))    
        mi_all.append(mutual_info/np.log(2))
        aleat_all.append(entropy_singlepass/np.log(2))
        # loss_all.append(loss.item())
    
    fr1_start = 0 #{0}
    fr1_end = fr1 #{49, 68}
    fr2_start = fr1 #49, 68
    fr2_end = len(indices) #len(val_indices) #{104, 145}
    
    # print(fr1_start, fr1_end, fr2_start, fr2_end)
  
    n_bins = 8
    
    uce_pe  = calibration(path_out, np.array(error_all), np.array(entropy_all), n_bins, x_label = 'predictive entropy')
    # print("Predictive Entropy")
    # print("uce = ", np.round(uce_pe, 2))

    uce_mi  = calibration(path_out, np.array(error_all), np.array(mi_all), n_bins, x_label = 'mutual information')
    # print("Mutual Information")
    # print("uce = ", np.round(uce_mi, 2))

    
    uce_ae  = calibration(path_out, np.array(error_all), np.array(aleat_all), n_bins, x_label = 'average entropy')
    # print("Average Entropy")
    # print("uce = ", np.round(uce_ae, 2))

    print("mean and std of error")
    print(error_all)
    print(np.mean(error_all)*100)
    print(np.std(error_all))
    mean_err_all = np.mean(error_all)*100
    sts_err_all = np.std(error_all)

    print("Average of expected error")
    print((np.array(avg_error_mean)).mean()*100)
    print((np.array(avg_error_mean)).std())

    mean_expected_error = np.array(avg_error_mean).mean()*100 
    std_expected_error = np.array(avg_error_mean).std()

    return mean_expected_error, std_expected_error, uce_pe, uce_mi, uce_ae


def get_logits(model, test_data_uncert, device, path, optimizer):
    test_data = get_test_data(test_data_uncert, path, device)
    indices = np.arange(0, len(test_data), 1)
    samples_iter = 200
    output_ = torch.zeros(samples_iter, len(test_data), 2)
    for index in indices:
        x = torch.unsqueeze(test_data[index][0].clone().detach(), 0) #torch.unsqueeze(torch.tensor(test_data[index][0]),0)
        #for a single datapoint
        with torch.no_grad():
            model.train(False)
            for j in range(samples_iter):
                with optimizer.sampled_params(train=False):                
                    x_test = x.to(device)
                    outputs = model(x_test)
                    output_[j][index] = outputs
    return output_