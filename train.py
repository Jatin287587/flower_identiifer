# Imports
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import json
import argparse

# Taking user input using ArgumentParser
parse= argparse.ArgumentParser(description='training NN')
parse.add_argument('--arch', type=str, default='vgg16')
parse.add_argument('--gpu', type=bool, default=True)
parse.add_argument('--lr', type=float, default=0.001)
parse.add_argument('--epochs', type=int, default=8)
parse.add_argument('--hidden_units', type=int, default=256)
parse.add_argument('--saved_model', type=str, default='checkpoint.pth')
parse.add_argument('--data_dir', type=str, default='flowers')
arg= parse.parse_args()

# Directories
data_dir = arg.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Defining transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(45),
                                      transforms.Resize(250),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(250),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(250),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Loading the datasets with ImageFolder
image_datasets= dict()
image_datasets['train'] = datasets.ImageFolder(train_dir, transform=train_transforms)
image_datasets['valid'] = datasets.ImageFolder(valid_dir, transform=valid_transforms)
image_datasets['test'] = datasets.ImageFolder(test_dir, transform=test_transforms)

# Using the image datasets and the trainforms, defining the dataloaders
dataloaders= dict()
dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
dataloaders['valid'] = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True)
dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True)

# TODO: Build and train your network
if arg.arch=='densenet':
    model= models.densenet121(pretrained=True)
    model_inp = 1024
elif arg.arch=='vgg13':
    model= models.vgg13(pretrained=True)
    model_inp = model.classifier[0].in_features
else:
    model= models.vgg16(pretrained=True)
    model_inp = model.classifier[0].in_features

for p in model.parameters():
    p.requires_grad= False

# Setting up Layers
classifier= nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(model_inp, 1024)),
    ('relu1', nn.ReLU()),
    ('dropout', nn.Dropout(p=0.5)),
    ('hidden1', nn.Linear(1024, 512)),
    ('relu2', nn.ReLU()),
    ('hidden2', nn.Linear(512, arg.hidden_units)),
    ('fc2', nn.Linear(arg.hidden_units, 102)),
    ('output', nn.LogSoftmax(dim=1))
]))

model.classifier= classifier

learnrate= arg.lr
epoch= arg.epochs
step= 0
print_every= 20

if arg.gpu:
    device='cuda'
else:
    device='cpu'

criterion= nn.NLLLoss()
optimizer= optim.Adam(model.classifier.parameters(), lr=learnrate)

model.to(device);

# Training Model
for e in range(epoch):
    running_loss= 0
    for ii, (image, label) in enumerate(dataloaders['train']):
        
        step+= 1
        image, label= image.to(device), label.to(device)
        
        optimizer.zero_grad()
        
        output= model.forward(image)
        loss= criterion(output, label)
        loss.backward()
        optimizer.step()
        
        running_loss+= loss.item()
        if step%print_every==0:
            total=0
            correct=0
            for data in dataloaders['valid']:
                model.eval()
                image_valid, label_valid= data
                image_valid, label_valid= image_valid.to(device), label_valid.to(device)
        
                output_valid= model(image_valid)
                _, pred= torch.max(output_valid.data, 1)
                loss_valid= criterion(output_valid, label_valid)
                loss_valid.backward()
                total+= label_valid.size(0)
                correct+= (pred==label_valid).sum().item()
            
            # Printing out Epoch, Training Loss, Validation Accuracy and Loss
            print('Epoch: {}/{}'.format(e+1, epoch))
            print('Training Loss:   ', loss.item()/print_every)
            print('Validation Accu: ', 100*(correct/total))
            print('Validation Loss: ', loss_valid.item()/print_every)
            print('_'*20)
            print('') # Left empty for output formatting

model.class_to_idx = image_datasets['train'].class_to_idx

# Setting up checkpoint
checkpoint= {'epoch': epoch,
            'optimizer_state': optimizer.state_dict,
            'state_dict': model.state_dict(),
            'model_inp': model_inp,
            'hidd_units': arg.hidden_units,
            'model_name': arg.arch,
            'class_to_idx': model.class_to_idx}

# Saving checkpoint
torch.save(checkpoint, 'checkpoint.pth')