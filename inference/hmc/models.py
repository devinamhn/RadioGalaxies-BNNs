import torch
import torch.nn as nn
import torch.nn.functional as F
#from layers import Linear_HMC
dropout_rate1= 0.0
dropout_rate2= 0.0
dropout_rate3= 0.0


class MLP(nn.Module):
    
    def __init__(self, input_size, hidden_size,output_size):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size*input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.imsize = input_size
        
        # weight initialisation:
        # following: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward (self, x):

        x = x.view(-1, self.imsize*self.imsize)
     
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)      
        
        x = self.out(x)

        #output = F.log_softmax(x, dim=1)
        return x

    def eval (self, input, target):
        out = self(input)
        return out

class LeNet(nn.Module):
    
    # def __init__(self, in_channels, output_size, jobid):
    def __init__(self, in_channels, output_size):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size = 5, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size = 5, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(16, 26, kernel_size = 5, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(26, 32, kernel_size = 5, stride = 1, padding = 1)

        self.fc1 = nn.Linear(7*7*32, 120)
        self.fc2 = nn.Linear(120,84)
        self.out = nn.Linear(84, output_size)
        #self.imsize = input_size
        
        # weight initialisation:
        # following: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # if(jobid==1):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            #     elif(jobid==2):
            #         nn.init.normal_(m.weight, 0, 0.1)
            #         nn.init.constant_(m.bias, 0)
            #     elif(jobid==3):
            #         nn.init.normal_(m.weight, 0, 1)
            #         nn.init.constant_(m.bias, 0)
                    
            # elif isinstance(m, nn.Conv2d):
            #     if(jobid==1):
            #         nn.init.normal_(m.weight, 0, 0.01)
            #         nn.init.constant_(m.bias, 0)
            #     elif(jobid==2):
            #         nn.init.normal_(m.weight, 0, 0.1)
            #         nn.init.constant_(m.bias, 0)
            #     elif(jobid==3):
            #         nn.init.normal_(m.weight, 0, 1)
            #         nn.init.constant_(m.bias, 0)

    def forward (self, x):

        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        
        x = torch.relu(self.conv4(x))
        x = torch.max_pool2d(x, 2)
        
        #print(x.shape)
        
        x = x.view(-1, 7*7*32)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.out(x)

        #output = F.log_softmax(x, dim=1)
        return x

    def eval (self, input, target):
        out = self(input)
        return out
    

class LeNetDrop(nn.Module):
    
    def __init__(self, in_channels, output_size, dropout_rate):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size = 5, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size = 5, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(16, 26, kernel_size = 5, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(26, 32, kernel_size = 5, stride = 1, padding = 1)

        self.fc1 = nn.Linear(7*7*32, 120)
        self.fc2 = nn.Linear(120,84)
        self.dropout1 = nn.Dropout(p = dropout_rate)

        self.out = nn.Linear(84, output_size)
        #self.imsize = input_size
        # weight initialisation:
        # following: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward (self, x):

        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        
        x = torch.relu(self.conv4(x))
        x = torch.max_pool2d(x, 2)
        
        #print(x.shape)
        
        x = x.view(-1, 7*7*32)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout1(x)
        x = self.out(x)

        #output = F.log_softmax(x, dim=1)
        return x

    def eval (self, input, target):
        out = self(input)
        return out