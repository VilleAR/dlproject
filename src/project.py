import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 
import shutil
#-----HYPERPARAMETERS
input_dim=16384 #128*128
hidden_dim=128
output_dim=14


#------DATA WRANGLING
train_transform = transforms.Compose([
                                        #transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])
dev_transform=transforms.Compose([transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])

shutil.copy('images/im1.jpg', 'data/train/im1.jpg')



