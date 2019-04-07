#!/usr/bin/env python3
#importing libraries
from torchvision import datasets, transforms, models
import torch
from PIL import Image  
from torch import nn
from torch import optim
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
import sys
class Ops:
    def make_trainingloader(dir):
        '''This function takes the directory of a folder which has two subfolders inside (train and valid)
        to create the datasets for training'''
        training_transforms = transforms.Compose([
                                        transforms.Resize(250),#1344
                                        transforms.RandomCrop(224),
                                        transforms.RandomRotation(90),
                                        transforms.RandomHorizontalFlip(p=0.50),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
        training_datasets = datasets.ImageFolder(dir, transform=training_transforms)
        trainloader = torch.utils.data.DataLoader(training_datasets, batch_size=64, shuffle=True)
        return trainloader, training_datasets.class_to_idx
    
    def make_network(in_size, out_size, hidden_layers, arch, droprate):
        '''This function instantiates a network and returns it'''
        model = getattr(models,arch)(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = Network(in_size, out_size,[hidden_layers], drop = droprate)
        return model
                     
    
    def make_validationloader(dir):
        '''This function takes the directory of a folder which has two subfolders inside (train and valid)
        to create the datasets for training'''
        validation_transforms = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], 
                                                       [0.229, 0.224, 0.225])])
        validation_datasets = datasets.ImageFolder(dir, transform=validation_transforms)
        validationloader = torch.utils.data.DataLoader(validation_datasets, batch_size=32)
        return validationloader
    
    def process_image(img_addr):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array

        '''
        image = Image.open(img_addr)
        means = np.array([0.485, 0.456, 0.406])
        sd = np.array([0.229, 0.224, 0.225])
        size = image.size
        ratio = None
        if size[0] < size[1]:
            ratio = 256/size[0]
            new_size = (256, int(size[1]*ratio))
        elif size[0] > size[1]:
            ratio = 256/size[1]
            new_size = (int(size[0]*ratio), 256)
        else:
            print('something went wrong')

        img = image.resize(new_size)
        img = np.array(img) / 255.0
        img_np = ((img - means) / sd)
        return img_np.transpose((2,0,1))



    


    def train_model(model, nepochs, state, lrate, trainloader, validationloader):
        if state:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                sys.exit("Unavle to use gpu")
        else:
            device="cpu"
        epochs = nepochs
        report_every = 20
        current_itr = 0
        round_loss = 0
        model.to(device)
        criterion = nn.NLLLoss() 
        optimizer = optim.Adam(model.classifier.parameters(), lr=lrate)
        for rounds in range(epochs):
            for i, (images, labels) in enumerate(trainloader):

                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                round_loss += loss.item()
                current_itr += 1
                if current_itr % report_every == 0:
                    model.eval()
                    with torch.no_grad():
                        t_loss = 0
                        acc = 0
                        for images, labels in validationloader:
                            images, labels = images.to(device), labels.to(device)
                            output = model.forward(images)
                            ps = torch.exp(output)
                            t_loss += criterion(output, labels).item()

                            matching = (labels.data == ps.max(dim=1)[1])
                            acc += matching.type(torch.FloatTensor).mean()
                    print("Epoch: {} of {}  ".format(rounds+1, epochs),
                          "Training Loss: {:.4f}  ".format(round_loss/report_every),
                          "Validation Loss: {:.3f}  ".format(t_loss/len(validationloader)),
                          "Validation Accuracy: {:.3f}".format(acc/len(validationloader)))
                    round_loss = 0
                    model.train()
        return model
        
class Network(nn.Module):
    def __init__(self, in_size, out_size, hidden_layers, drop=0.5):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_size,hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1],hidden_layers[1:])
        self.layers.extend([nn.Linear(l1, l2) for l1, l2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], out_size)
        self.dropout = nn.Dropout(p=drop)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.output(x)
        return F.log_softmax(x, dim=1)
        

