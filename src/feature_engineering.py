from sklearn.cluster import SpectralClustering
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def prepare_spectral_clustering_features(X, n_clusters):
    '''
    Inputs:
        X: data matrix or dataframe. Each data instance is expected to be a row
            in the matrix or dataframe
        n_cluster: number of clusters
    Outputs:
        return: returns a one-hot vector encoding of the clusters. For example, if 
        there are 6 data points, belonging to clusters [0,0,1,1,2,2], then the return
        array will be
        [1,0,0]
        [1,0,0]
        [0,1,0]
        [0,1,0]
        [0,0,1]
        [0,0,1]
    '''
    cluster_model = SpectralClustering(n_clusters=n_clusters,
                                       n_init = 10,
                                       assign_labels="discretize",
                                       random_state=0)
    cluster_model.fit(X)
    labels_vec = cluster_model.labels_
    labels_vec = np.reshape(labels_vec,(len(labels_vec),1),'F')
    enc = OneHotEncoder(handle_unknown='error')
    enc.fit(labels_vec)
    one_hot_vec = enc.transform(labels_vec)
    return one_hot_vec

def prepare_eigen_component_features(images_list,eig_vecs,):
    '''
    Inputs:
        images_list: list of images (images are ndarrays, usually of 2 dimensions (e.g 128x128))
        eig_vecs: an 2d array of eigenvectors, where each eigenvector is a column
    Outputs:
        return: returns a feature matrix, where the rows of the feature matrix correspond to the 
        features of images. First image is the first row of the matrix, last image is the last row.
        The features are the components of the eigenvectors (which were passed in the input 'eig_vecs')
    '''
    n_components = eig_vecs.shape[1]
    feature_matrix = np.zeros(len(images_list),n_components)
    for image in images_list:
        image_vec = unravel_image(image)
        components = eig_vecs.T@image_vec
        feature_matrix[ii,:] = components
    return feature_matrix

    
    

    