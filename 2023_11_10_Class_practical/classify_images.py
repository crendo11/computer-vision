"""
This scripts compares each feature of each image with the computed k-means centroids
and creates a histogram of the features of each image
"""

import numpy as np
import json
from numba import jit
import sys
import matplotlib.pyplot as plt


# adding Folder_2/subfolder to the system path
sys.path.insert(0, './')
import k_means_gpu as km

# Load the HOG descriptors from the JSON file
hog_descriptors = []
with open('./2023_11_10_Class_practical/features_t_05.txt', 'r') as file:
    for line in file:
        # Parse the JSON string from the line
        entry = json.loads(line)
        
        # Convert lists to NumPy arrays if needed
        # entry["hog_descriptors"] = [np.array(descriptor) for descriptor in entry["hog_descriptors"]]
        
        # Append the entry to the list
        hog_descriptors.append(entry)


# load the centroids
centroids = np.loadtxt('./2023_11_10_Class_practical/k_means_t_05.txt')

@jit(nopython=True)
def calc_histrogram(features, centroids):
    # initialize the histogram
    histogram = np.zeros(64)
    # iterate over the features
    distances = np.zeros((len(features), len(centroids)), dtype=np.float64)
    for i in range(len(features)):
        for k in range(len(centroids)):
            # calculate the distance between the feature and the centroids
            distances[i][k] = km.distance(features[i], centroids[k])
        # get the index of the minimum distance
        index = np.argmax(distances[i])
        # increment the histogram at the index
        histogram[index] += 1
    # normalize the histogram manually
    norm = np.sqrt(np.sum(histogram ** 2))
    histogram = histogram / norm
    return histogram

centroids = np.array(centroids, dtype=np.float64)

# iterate over the images and calculate the histogram of each image
for i in range(len(hog_descriptors)):
    # get the features of the image
    hog_descriptor = hog_descriptors[i]
    # calculate the histogram of the features of the image
    features =np.array(hog_descriptor['features'], dtype=np.float64)
    histogram = calc_histrogram(features, centroids)
    # add the histogram to the dictionary
    hog_descriptor['histogram'] = histogram.tolist()
    hog_descriptors[i] = hog_descriptor

    print('Image {} processed.'.format(i))

    # plot the histogram as line plot
    plt.figure()
    plt.bar(np.arange(len(histogram)), histogram)


plt.show()

# Save the HOG descriptors to a JSON lines file
with open('./2023_11_10_Class_practical/classified_images_t_05.txt', 'w') as file:
    for entry in hog_descriptors:
        # Convert the entry to a JSON string and write it to a line in the file
        file.write(json.dumps(entry) + '\n')
