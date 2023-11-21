import numpy as np
import json
import sys

# adding Folder_2/subfolder to the system path
sys.path.insert(0, './')
import k_means_gpu as km


hog_descriptors = []
# Load the HOG descriptors from the JSON file
with open('./2023_11_10_Class_practical/features_t_05.txt', 'r') as file:
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
centroids = km.read_centers('./2023_11_10_Class_practical/k_means.txt')


# initialize the distances
distances = np.ones((num_features, k))

centroids = np.array(centroids, dtype=np.float64)
centroids, labels, distances, loss = km.k_means(features, centroids, labels, distances, iterations)

# write the k centroids to a file
with open('./2023_11_10_Class_practical/k_means_t_05_loss_{}.txt'.format(loss), 'w') as file:
    for centroid in centroids:
        file.write('{}\n'.format(centroid))