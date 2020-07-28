import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from sklearn import metrics
from skimage import io, color

from src.utils import unravel_image

import time
import os
import pickle



def train_valid(optimizer = None,
                epochs = 20,
                model = None,
                train_criterion = None,
                train_loader = None,
                valid_criterion = None,
                valid_loader = None,
                device = None,
                model_output_path = '.',
                with_features=False,
                train_eig_vecs = None,
                ):

    start_epoch = 1
    #or: best_val_acc = 0
    best_val_loss = np.inf

    history = {"train_loss":[], "train_acc":[],
                "valid_loss":[], "valid_acc":[], "valid_preds_list":[],
                "valid_truelabels_list":[], "valid_probas_list":[], "valid_auc_score":[]}

    start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):

        train_loss, train_acc = train(epoch, model, optimizer, train_criterion, 
                                      train_loader, device, with_features, train_eig_vecs)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        print('epoch: ', epoch)
        print('{}: loss: {:.4f} acc: {:.4f}'.format('training', train_loss, train_acc))

        valid_loss, valid_acc, valid_preds_list, valid_truelabels_list, valid_probas_list, valid_auc_score = validation(epoch, model, optimizer, 
                                                        valid_criterion, valid_loader, 
                                                        device, with_features, train_eig_vecs)
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
            best_model_file = model_output_path + "best_run_param.pth"
            torch.save(model.state_dict(), best_model_file)

        # save model from every training epoch
        # can be deleted if do not need this one, or adapt it to save 5th, 10th, 15th ...models
        model_file = model_output_path+"/param_" + str(epoch) + ".pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            }, model_file)
        # save training/validation results
        with open(model_output_path+"history.pkl", "wb") as fout:
            pickle.dump(history, fout)
        print('time elapsed:', time.time() - start_time)
    print('total time elapsed:', time.time() - start_time)
 
    return history

def train(epoch,
          model,
          optimizer, 
          criterion, 
          loader, 
          device, 
          with_features=False, 
          train_eig_vecs=None,
          ):

    model.train()

    running_loss = 0.0
    epoch_loss = 0.0
    total_samples = 0
    correct = 0
    mysoftmax = nn.Softmax(dim=1)

    for batch_idx, samples in enumerate(loader):

        image = samples[0].to(device)
        label = samples[1].to(device)
        #print(image)
        batch_size = image.size(0)
        if with_features:
            feat = prepare_eigen_component_features(image, train_eig_vecs,device).to(device)
            output = model(image,feat)
        else:
            output = model(image)
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

def validation(epoch, 
               model, 
               optimizer, 
               criterion, 
               loader, 
               device, 
               with_features = False,
               train_eig_vecs=None, #yes, train! 
               multiclass=True,
               ):

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
            label = samples[1].to(device)
            batch_size = image.size(0)
            if with_features:
                #print(image.shape)
                feat = prepare_eigen_component_features(image, train_eig_vecs,device).to(device)
                output = model(image,feat)
            else:
                output = model(image)
            
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

def prepare_eigen_component_features(images_tensor,eig_images,device):
    '''
    Inputs:
        images_tensor: batch tensor of images (images are tensors of 
        shape [batch_size, 1, pixel_1,pixel_2]). "1" represents grayscale
        eigen_images: p-length list of two dimensional arrays of size m x n, 
            each representing an eigen image. Note that this is not an "input"
            into this function (look in the function definition), but this 
            variable is hardcoded into the function via create_function_helper
    Outputs:
        return: returns a feature matrix, where the rows of the feature matrix correspond to the 
        features of images. First image is the first row of the matrix, last image is the last row.
        The features are the components of the eigenvectors (which were passed in the input 'eig_vecs')
    '''
    images_tensor.to(device)
    list_eig_coefs = list(
                        map(create_function_helper(eig_images),
                            torch.unbind(images_tensor, 0)
                           )
                        )
    features_tensor = torch.stack(list_eig_coefs)
    #print(features_tensor.shape) #should be of torch.Size([p])
    return features_tensor

def create_function_helper(eigen_images):
    '''
    Inputs:
    eigen_images: p-length list of two dimensional arrays of size m x n, 
            each representing an eigen image. Note that this is not an "input"
            into this function (look in the function definition), but this 
            variable is hardcoded into the function via create_function_helper
    This function returns another function. The output function is used in a
    map() call, which takes in a function and an iterable, and applies the 
    function to each object in the iterable.

    Our map function would like to take two inputs, the first being the image
    to project onto the eigenvectors, and the second being the actual eigenvectors

    Since our mapped function can only have one input, so we pass in another 
    input to the function using this helper method, which returns a function 
    that already has the eigen_images pre-defined
    '''
    def project_image_onto_eigen_pytorch_(image_tensor):
        '''
        Inputs:
            image_tensor: a two dimensional tensor of size m x n, representing 
            one single image
            eigen_images: p-length list of two dimensional arrays of size m x n, 
            each representing an eigen image. Note that this is not an "input"
            into this function (look in the function definition), but this 
            variable is hardcoded into the function via create_function_helper
        Outputs:
            eig_coefs_tensor: a 1 dimensional tensor with p entries, each
            entry represents a coefficient of the image projected onto the 
            eigenvectors
        '''
        #unravel image_tensor 
        image_vec_tensor = unravel_image_to_tensor(image_tensor[0].data.cpu().numpy())
        #print(image_vec_tensor.shape) #should be of torch.Size([1, p*m])
        #unwrap each eigen_image
        list_eig_vec_tensor =  list(map(unravel_image_to_tensor,eigen_images))
        eig_vecs_tensor = torch.stack(list_eig_vec_tensor) 
        #print(eig_vecs_tensor.shape) # should be torch,Size([p, (n*m)])
        eig_coefs_tensor = torch.squeeze(
                               torch.matmul(
                                   eig_vecs_tensor,
                                   torch.transpose(image_vec_tensor,0,1)
                                )
                            )
        #print(eig_coefs_tensor.shape) # should be of torch.size(p)
        return eig_coefs_tensor
    return project_image_onto_eigen_pytorch_
def unravel_image_to_tensor(image):
    return torch.from_numpy(unravel_image(image)).float()

