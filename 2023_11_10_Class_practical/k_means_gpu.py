"""
 k menans algorith to classify image features into 64 classes
 each feature is a vector of 4 elements
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

hog_descriptors = []
# Load the HOG descriptors from the JSON file
with open('./2023_11_10_Class_practical/features_t_5.txt', 'r') as file:
    for line in file:
        # Parse the JSON string from the line
        entry = json.loads(line)
        
        # Convert lists to NumPy arrays if needed
        # entry["hog_descriptors"] = [np.array(descriptor) for descriptor in entry["hog_descriptors"]]
        
        # Append the entry to the list
        hog_descriptors.append(entry)

# define the constants
k = 64
iterations = 100

features = []
# create and array to store all the features from all the images
for i in range(len(hog_descriptors)):
    # convert the list of features to a numpy array
    hog_descriptors[i]['features'] = np.array(hog_descriptors[i]['features'])
    # append the features to the list of features
    features.extend(hog_descriptors[i]['features'])

features = np.array(features)
num_features = len(features)
num_images = len(hog_descriptors)
labels = np.zeros(num_features)

# initialize centroids with k random features from the features
# centroids = features[np.random.choice(num_features, k, replace=False)]
centroids = read_centers('./2023_11_10_Class_practical/k_means.txt')


# initialize the distances
distances = np.ones((num_features, k))

@jit(nopython=True)
def k_means(features, centroids, labels, distances, iterations):
    loss = 1e10
    # iterate over the number of iterations or when the loss is 0
    while iterations > 0 or loss == 0:
        # initialize the loss (number of misclassified features)
        loss = 0
        # iterate over the features
        for i in range(num_features):
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

        features_j = np.empty_like(features)  # pre-allocate features_j
        # iterate over the centroids
        for j in range(k):
            # get the features with the label j
            indices = [i for i, x in enumerate(labels) if x == j]
            for i in indices:
                features_j[i] = features[i]
            # calculate the mean of the features manually
            centroids[j] = np.sum(features_j, axis=0) / features_j.shape[0]


        # update the number of iterations
        iterations -= 1
        print('Loss: ' + str(loss))
        print('Iterations: ' + str(iterations))

    return centroids, labels, distances, loss

centroids = np.array(centroids, dtype=np.float64)
centroids, labels, distances, loss = k_means(features, centroids, labels, distances, iterations)

# write the k centroids to a file
with open('./2023_11_10_Class_practical/k_means_loss_{}.txt'.format(loss), 'w') as file:
    for centroid in centroids:
        file.write('{}\n'.format(centroid))