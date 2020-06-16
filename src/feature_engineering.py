from sklearn.cluster import SpectralClustering
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def prepare_spectral_clustering_features(X, n_clusters):
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
    n_components = eig_vecs.shape[1]
    feature_matrix = np.zeros(len(images_list),n_components)
    for image in images_list:
        image_vec = unravel_image(image)
        components = eig_vecs.T@image_vec
        feature_matrix[ii,:] = components
    return feature_matrix

    
    

    