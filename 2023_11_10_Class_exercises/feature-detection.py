import cv2
import matplotlib.pyplot as plt
import numpy as np

# load the image
image = cv2.imread('./2023_11_10/images/1.jpg')
# convert to gray scale image
Iog = image.copy()
I = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# define the size of the window
u = 3
v = 3

# constants
k = 0.04
sigma = 0.5
threshold = 0.01

def calcGradient(M):
    # create the gradient matrix
    Ix = np.zeros(M.shape)
    Iy = np.zeros(M.shape)
    # for i in range(M.shape[0] - 1):
    #     for j in range(M.shape[1] - 1):
    #         # caculate the gradient manually
    #         Ix[i][j] = I[i+1][j] - I[i][j]
    #         Iy[i][j] = I[i][j+1] - I[i][j]
    # calculate the gradient using the sobel filter
    Ix = cv2.Sobel(M, cv2.CV_64F, 1, 0)
    Iy = cv2.Sobel(M, cv2.CV_64F, 0, 1)

    return Ix, Iy

# create w matrix
w = np.zeros((u,v))
for i in range(u):
    for j in range(v):
        w[i][j] = np.exp(-(i+j)**2/(2*sigma))

# apply the gaussian filter to the image
for i in range(I.shape[0] - u):
    for j in range(I.shape[1] - v):
        I[i][j] = np.sum(np.multiply(I[i:i+u, j:j+v], w))

# create the harris matrix
H = np.zeros(I.shape)

# get the gradients
Ix, Iy = calcGradient(I)
# calculate the harris matrix
for i in range(I.shape[0] - u):
    for j in range(I.shape[1] - v):
        # calculate the elements of the matrix M
        A = Ix[i][j]**2
        B = Iy[i][j]**2
        C = Ix[i][j]*Iy[i][j]
        det = A*B - C**2
        # # calculate the harris matrix
        H[i][j] = det - k*((A + B)**2)
        # mark the coreners 
        if H[i][j] > threshold:
            cv2.circle(Iog, (j,i), 100, (0,0,255), 1)

# show the image with matplotlib
plt.imshow(Iog)
plt.show()