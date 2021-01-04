
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as f
import torchvision
from torchvision import datasets, transforms, models# Imports here
import PIL
from PIL import Image
import glob, os
import random
import numpy as np
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--arch',
                    action='store',
                    default='vgg16',
                    dest='arch_value',
                    help='action to specify model architecture')
parser.add_argument('file_name', 
                    action='store',
                    help='specify filepath')
parser.add_argument('--save_dir',
                    action='store',
                    default='my_checkpoint.pth',
                    dest='save_path',
                    help='specified path to save model')
parser.add_argument('--learning_rate',
                    action='store',
                    default=0.001,
                    type=float,
                    dest='learning_rate',
                    help='specifies learning rate')
parser.add_argument('--epochs',
                    action='store',
                    default= 2,
                    type=int,
                    dest='epochs',
                    help='sets the amount of epochs for training')
parser.add_argument('--hidden_units',
                    action='store',
                    default=4096,
                    type=int,
                    dest='hidden_units',
                    help='sets hidden unit value')
parser.add_argument('--gpu',
                    action='store_true',                    
                    dest='cuda',
                    help='set to use gpu as processor')
                    
result = parser.parse_args()
print('cuda: {}'.format(result.cuda))


data_dir = result.file_name
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
                    'valid': transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
                    'test': transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid', 'test']} 
dataLoaders = {x: torch.utils. data.DataLoader(image_datasets[x], batch_size=64, shuffle = True) for x in ['train', 'valid', 'test']}

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
import torchvision.models as models
if result.arch_value == "vgg13":
    model = models.vgg13(pretrained = True)
elif result.arch_value == "vgg16":
    model = models.vgg16(pretrained = True)
else:
    print ("Choose model architecture, vgg16 or vgg13")
    exit()
    


for param in model.parameters():
    param.requires_grad = False
    
    
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, result.hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(result.hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim = 1))]))
model.classifier = classifier

import time

for device in ['cuda']:
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=result.learning_rate)
    
    model.to(device)
    
    for ii, (inputs, labels) in enumerate(dataLoaders['train']):
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        start = time.time()
        
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if ii == 3:
            break
            
    print(f'Device is: {device}; Time taken each readthrough is: {(time.time() - start)/3:.3f} seconds')
    
device = torch.device("cuda" if torch.cuda.is_available() and result.cuda else 'cpu')



for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(25088, result.hidden_units), 
                                nn.ReLU(), 
                                nn.Dropout(p=0.5), 
                                nn.Linear(result.hidden_units, 102), 
                                nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=result.learning_rate)
                 
model.to(device)

epochs = result.epochs
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []
for epoch in range(epochs):
    for inputs, labels in dataLoaders['train']:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0 
            model.eval()
            with torch.no_grad():
                for inputs, labels in dataLoaders['test']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train Loss: {running_loss/print_every:.3f}.. "
                      f"validation Loss: {test_loss/len(dataLoaders['valid']):.3f}.. "
                      f"validation accuracy: {accuracy/len(dataLoaders['valid']):.3f}")
                    
                running_loss = 0
                model.train()
                train_losses.append(running_loss/len(dataLoaders['train']))
                test_losses.append(test_loss/len(dataLoaders['test']))
                
                
                
#plt.plot(train_losses, label='Training loss')
#plt.plot(test_losses, label='Testing loss')
#plt.legend(frameon=False)


print("Accuracy is: {}".format(accuracy/len(dataLoaders['valid'])*100))
model_final = None
if result.cuda and torch.cuda.is_available():
    model_final = model.cuda()
else:
    model_final = model.cpu()

    
model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'model': model_final,
              'mapping': model.class_to_idx,
              'features': model.features}


torch.save(checkpoint, result.save_path)
