import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from sklearn import metrics

from skimage import io, color

import time
import os
import pickle

import matplotlib.pyplot as plt

def train(epoch, model, optimizer, criterion, loader, device, train_feat):

    model.train()

    running_loss = 0.0
    epoch_loss = 0.0
    total_samples = 0
    correct = 0
    mysoftmax = nn.Softmax(dim=1)

    for batch_idx, samples in enumerate(loader):

        image = samples[0].to(device)
       
        #label = samples[1].squeeze().to(device)
        label = samples[1].to(device)
        
        #label = torch.tensor(label, dtype=torch.long, device=device)
        #print(image)
        batch_size = image.size(0)
        feat = torch.add(torch.ones([batch_size,1]),train_feat.flatten()).to(device)
        output = model(image,feat)
        _, preds = torch.max(output, dim = 1)
    
        loss = criterion(output, label)
        running_loss += loss.item()
        epoch_loss += loss.item()

        total_samples += image.shape[0]
        correct += torch.sum(preds == label).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss = 0.0

    training_accuracy = correct / total_samples

    return epoch_loss / len(loader), training_accuracy

def validation(epoch, model, optimizer, criterion, loader, device, valid_feat, multiclass=True):

    model.eval( )

    running_loss = 0.0
    total_samples = 0
    correct = 0
    mysoftmax = nn.Softmax(dim=1)

    # this part needs to be adapted later based on task 3
    preds_list = []
    truelabels_list = []
    probas_list = []

    with torch.no_grad():
        for batch_idx, samples in enumerate(loader):

            image = samples[0].to(device)
            
            # label = samples[1].squeeze().to(device)
            #label = torch.tensor(label, dtype=torch.long, device=device)
            label = samples[1].to(device)
            batch_size = image.size(0)
            feat = torch.add(torch.ones([batch_size,1]),valid_feat.flatten()).to(device)
            output = model(image,feat)
            output_softmax = mysoftmax(output)

            _, preds = torch.max(output, dim = 1)

            loss = criterion(output, label)
            running_loss += loss.item()

            total_samples += image.shape[0]
            correct += torch.sum(preds == label).item()

            preds_list.append(preds.cpu().numpy())
            truelabels_list.append(label.cpu().numpy())
            probas_list.append(output_softmax.cpu().numpy())
    
        valid_accuracy = correct / total_samples

        probas_list = np.vstack(probas_list)
        truelabels_list = np.concatenate(truelabels_list)
        preds_list = np.concatenate(preds_list)
        if multiclass == False:
            auc_score = metrics.roc_auc_score(truelabels_list, preds_list)

        else:
            # Computes the average AUC of all possible pairwise combinations of classes
            # Insensitive to class imbalance when average == 'macro'
            auc_score = metrics.roc_auc_score(truelabels_list, probas_list, multi_class='ovo')


        return running_loss / len(loader), valid_accuracy, preds_list, truelabels_list, probas_list, auc_score
