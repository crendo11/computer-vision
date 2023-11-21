"""
 k menans algorith to classify image features into 64 classes
 each feature is a vector of 4 elements
 Optimized to work with GPU
"""
import numpy as np
import json
from numba import jit


# define distance function
@jit(nopython=True)
def distance(x, y):
    # calculate the norm of the vectors
    x_norm = np.sqrt(np.sum(x ** 2))
    y_norm = np.sqrt(np.sum(y ** 2))
    # normalize the vectors
    x = x / x_norm
    y = y / y_norm
    # calculate the dot product manually
    dot_product = np.sum(x * y)
    return dot_product

def read_centers(filename):
    centers = np.loadtxt(filename, dtype=float)
    return centers

@jit(nopython=True)
def k_means(features, centroids, labels, distances, iterations):
    loss = 1e10
    k = centroids.shape[0]
    # iterate over the number of iterations or when the loss is 0
    while iterations > 0 and loss > 0:
        # initialize the loss (number of misclassified features)
        loss = 0
        # iterate over the features
        for i in range(len(features)):
            # iterate over the centroids
            for j in range(k):
                # calculate the distance between the feature and the centroid
                distances[i][j] = distance(features[i], centroids[j])
                # get the index of the minimum distance
            index = np.argmax(distances[i])
            if labels[i] != index:
                loss += 1
            # assign the label to the feature
            labels[i] = index


        # iterate over the centroids
        for j in range(k):
            # get the features with the label j
            indices = np.argwhere(labels == j)
            features_j = np.empty_like(features)  # pre-allocate features_j
            h = 0
            for i in indices:
                features_j[h] = features[i]
                h += 1
            # calculate the mean of the features manually
            centroids[j] = np.sum(features_j, axis=0) / features_j.shape[0]


        # update the number of iterations
        iterations -= 1
        print('Loss: ' + str(loss))
        print('Iterations: ' + str(iterations))

    return centroids, labels, distances, loss

