import matplotlib.pyplot as plt
import numpy as np
import cv2 
import os
from skimage.color import rgb2gray
import pickle

def plot_image_grid(images, 
                    title, 
                    image_shape=(64,64),
                    ncols=5,
                    nrows=2, 
                    bycol=0, 
                    row_titles=None,
                    col_titles=None,
                    save=False):
    fig,axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(2. * n_col, 2.26 * n_row))
    for i, image in enumerate(images):
        row,col = reversed(divmod(i,n_row)) if bycol else divmod(i,n_col) 
        if nrow==1:
            cax = axes[col]
        else:
            cax = axes[row,col]
        cax.imshow(image.reshape(image_shape), cmap='gray',
                   interpolation='nearest',
                   vmin=image.min(), vmax=image.max())
        cax.set_xticks(())
        cax.set_yticks(())
    if row_titles is not None :
        for ax,row in zip(axes[:,0],row_titles) :
            ax.set_ylabel(row,size='large')
    if col_titles is not None :
        for ax,col in zip(axes[0],col_titles) :
            ax.set_title(col)
    
    fig.suptitle(title)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    if save is True:
        plt.savefig(title + '.pdf',bbox_inches='tight')
    plt.show()


def unravel_image(image):
    '''
    Unravels or flattens an image by taking its columns and appends each column
    to the bottom of the first vector.
    E.G, if an image looks like
    1 4 7
    2 5 8
    3 6 9
    Then the vector will be [1,2,3,4,5,6,7,8,9] (Transposed)
    '''
    num_pixels = image.shape[0]*image.shape[1]
    image_vector = np.reshape(image,(-1,num_pixels))
    return image_vector

def ravel_image_vec(image_vector,image_dim):
    '''
    Ravels or 'unflattens' a vector into an image by filling up the image by 
    columns first, then rows.
    E.g, [1,2,3,4,5,6,7,8,9,10,11,12], (4,3) goes to
    1 5 9
    2 6 10
    3 7 11
    4 8 12
    '''
    image = np.reshape(image_vector,image_dim,'F')
    image = image.T
    return image

def make_pyramids(images_list,num_levels):
    '''
    return struct: list_1 of list_2 of arrays
    list_2 has length equal to num_levels, and contains num_levels amount of pyramids
    list_1 has length equal to len(images_list)
    
    Access the return by all_pyramids_list[image_index][level_index]
    '''
    all_pyramids_list = []
    for image in images_list:
        prev_level = image
        one_pyramid_list = [image]
        for ii in range(num_levels):
            im_down = cv2.pyrDown(prev_level)
            prev_level = im_down
            one_pyramid_list.append(im_down)
        all_pyramids_list.append(one_pyramid_list)
    return all_pyramids_list

def PCA_images_list(images_list):
    #Initialize X
    default_image_size = images_list[0].shape
    num_pixels = images_list[0].shape[0]*images_list[0].shape[1]
    X = np.zeros((len(images_list),num_pixels))
    #Now grab all the images and push them into X
    for ii,image in enumerate(images_list):
        #We "unravel" the image into a 1d array, we will have to "ravel" it later
        if image.shape!= default_image_size:
            raise ValueError('All images must be same size. Image number '+ str(ii +1)+
                             'was not the same size as the first image in the list')
        X[ii,:] = unravel_image(image)
            
    #Perform PCA on X
    cov = (X-X.mean(axis=0)).T@(X-X.mean(axis=0))
    eig_vals,eig_vecs = np.linalg.eigh(cov)
     
    #"ravel" each eigenvector and push into new_eigvec_list
    new_eigvec_list = []
    for ii in range(len(eig_vecs)):
        
        new_eig_vec = ravel_image_vec(eig_vecs[:,ii],default_image_size)
            
        new_eigvec_list.append(new_eig_vec)
    #Now, return the eigval-eigvec pairs
    return (eig_vals,new_eigvec_list)

def PCA_pyramids(all_pyramids_list):
    '''
    For each level in each image pyramid, do a PCA on it, and return the eigenvalues and eigenvectors for
    each level in the pyramid.
    '''
    num_pyramids = len(all_pyramids_list)
    num_levels = len(all_pyramids_list[0])
    eig_vals_vecs_per_level = [] # This is our return variable
    
    #First Loop, we group the images over each level
    for level_idx in range(num_levels):
        #Initialize X
        #Now grab all the images and push them into images_list
        image_list = []
        for img_idx in range(num_pyramids):
            image = all_pyramids_list[img_idx][level_idx]
            image_list.append(image)
        
        eig_vals_vecs_pair = PCA_images_list(image_list)
        eig_vals_vecs_per_level.append(eig_vals_vecs_pair)
        
    return eig_vals_vecs_per_level
