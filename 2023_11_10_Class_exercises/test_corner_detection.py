import cv2
import matplotlib.pyplot as plt
import numpy as np

# load the image
image = cv2.imread('./2023_11_10_Class_exercises/images/1.jpg')
# convert to gray scale image
Iog = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Sobel kernels
Sx = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]])

Sy = Sx.T

# Gaussian Kernel
G = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]])/16


def corner_response(image, k, Sx=Sx, Sy=Sy, G=G):
    # compute first derivatives
    dx = cv2.filter2D(image, ddepth=-1, kernel=Sx)
    dy = cv2.filter2D(image, ddepth=-1, kernel=Sy)

    # Gaussian Filter (blur)
    A = cv2.filter2D(dx*dx, ddepth=-1, kernel=G)
    B = cv2.filter2D(dy*dy, ddepth=-1, kernel=G)
    C = cv2.filter2D(dx*dy, ddepth=-1, kernel=G)

    # compute corner response at all pixels
    return (A*B - (C*C)) - k*(A + B)*(A + B)

# hyperparameters
k = 0.05
thresh = 0.5

# thresholded corner responses
#strong_corners = corner_response(image, k) > thresh

def get_harris_corners(image, k=k,Sx=Sx, Sy=Sy, G=G):

    # compute corner response
    R = corner_response(image, k, Sx=Sx, Sy=Sy, G=G)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(R > 1e-2))
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    return cv2.cornerSubPix(image, np.float32(centroids), (9,9), (-1,-1), criteria)

# convert image to float32
image = np.float32(image)
# 0-1 normalize
image /= image.max()

# find corners
corners = get_harris_corners(image)

# draw corners on output image
image_out = np.dstack((image, image, image))
for (x, y) in corners:
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)
    cv2.circle(image_out, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

# display image
plt.figure(figsize=(10, 10))
plt.imshow(image_out)
plt.axis('off')
plt.show()