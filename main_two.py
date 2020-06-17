###################################
############# Notice ##############
###################################

# Some parts need to be adapted later

###################################
########## import library #########
###################################

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

#from skimage import io, color

import time
import os
import pickle

import matplotlib.pyplot as plt
import scikitplot as skplt

from train_valid import train, validation
from Basic_CNN_Architecture import BasicCNN


###################################
########### load device ###########
###################################

# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


###################################
############ def path #############
###################################

# hpc
# please change file path here
train_df_path =
val_df_path =
test_df_path =

root_dir =




###################################
############ load data ############
###################################

# after feature engineering(task 3)

train_loader =
valid_loader =
test_loader =


###################################
############ Train ############
###################################
print('###################################')
print('############## Train ##############')
print('###################################')


# training process
# to be finished later
model = BasicCNN(in_channels,
			     enc_channels,
			     out_channels,
			     lin_channels,
			     num_classes,
			     kernel_size,
			     stride,
			     dropout = None,
			     activation = nn.ReLU(inplace = False))
model.to(device)

def train_valid(optimizer = optim.Adam(model.parameters()), epochs = 20, model = model,
                train_criterion = nn.CrossEntropyLoss(), train_loader = train_loader,
                valid_criterion = nn.CrossEntropyLoss(), valid_loader = valid_loader,
                device = device):

    start_epoch = 1
    #or: best_val_acc = 0
    best_val_loss = np.inf

    history = {"train_loss":[], "train_acc":[],
                "valid_loss":[], "valid_acc":[], "valid_preds_list":[],
                "valid_truelabels_list":[], "valid_probas_list":[], "valid_auc_score":[]}

    start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):

        train_loss, train_acc = train(epoch, model, optimizer, train_criterion, train_loader, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        print('epoch: ', epoch)
        print('{}: loss: {:.4f} acc: {:.4f}'.format('training', train_loss, train_acc))

        valid_loss, valid_acc, valid_preds_list, valid_truelabels_list, valid_probas_list, valid_auc_score = validation(epoch, model, optimizer, valid_criterion, valid_loader, device)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)
        history["valid_preds_list"].append(valid_preds_list)
        history["valid_truelabels_list"].append(valid_truelabels_list)
        history["valid_probas_list"].append(valid_probas_list)
        history["valid_auc_score"].append(valid_auc_score)

        print('{}: loss: {:.4f} acc: {:.4f} auc: {:.4f}'.format('validation', valid_loss, valid_acc, valid_auc_score))
        print()

        # save models(use valid loss as best model criterion, please change
        # criterion here if needed(eg. valid acc)
        is_best = valid_loss < best_val_loss
        best_val_loss = min(valid_loss, best_val_loss)

        if is_best:
            # please change model file path here
            best_model_file = "best_models/best_model.pth"
            torch.save(model.state_dict(), best_model_file)

        # save model from every training epoch
        # can be deleted if do not need this one, or adapt it to save 5th, 10th, 15th ...models
        model_file = "models/resnet152_model_" + str(epoch) + ".pth"

        torch.save(model.state_dict(), model_file)

        # save training/validation results
        with open("history.pkl", "wb") as fout:
            pickle.dump(history, fout)

    print('time elapsed:', time.time() - start_time)

    return history

results = train_valid()




###################################
############## Test ###############
###################################
print('###################################')
print('############# Test ################')
print('###################################')

def test(optimizer = optim.Adam(model.parameters()), model = model,  test_criterion = nn.CrossEntropyLoss(),
         loader = test_loader, device = device):

    model.eval()

    running_loss = 0.0
    total_samples = 0
    correct = 0
    mysoftmax = nn.Softmax(dim=1)

    preds_list = []
    truelabels_list = []
    probas_list = []
    with torch.no_grad():
        for batch_idx, samples in enumerate(loader):

            image = samples['image'].to(device)
            label = samples['label'].squeeze()
            label = torch.tensor(label, dtype=torch.long, device=device)

            output = model(image)
            output_softmax = mysoftmax(output)

            _, preds = torch.max(output, dim = 1)

            loss = test_criterion(output, label)
            running_loss += loss.item()

            total_samples += image.shape[0]
            correct += torch.sum(preds == label).item()


            preds_list.append(preds.cpu().numpy())
            truelabels_list.append(label.cpu().numpy())
            probas_list.append(output_softmax.cpu().numpy())

        test_accuracy = correct / total_samples

        return running_loss / len(loader), test_accuracy, preds_list, truelabels_list, probas_list


test_loss, test_acc,  preds_list, truelabels_list, probas_list= test()