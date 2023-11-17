import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_harris_corners(I, k, Gx, Gy, w):
    # normalize the image
    I = np.float32(I)
    I /= I.max()

    # compute the gradients
    Ix, Iy = calcGradient(I, Gx, Gy)

    # create matrices of A, B, C and det
    # create matrices of A, B, C and det
    A = cv2.filter2D(Ix*Ix, ddepth=-1, kernel=w)
    B = cv2.filter2D(Iy*Iy, ddepth=-1, kernel=w)
    C = cv2.filter2D(Ix*Iy, ddepth=-1, kernel=w)
    det = A*B - C*C

    # calculate the determinant and trace
    det = A*B - C*C
    H = det - k*(A + B)**2

    return H


def calcGradient(M, kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]]), ky = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])):
    # use 2d filter function to calculate the gradient
    Ix = cv2.filter2D(M, ddepth=-1, kernel=kx)
    Iy = cv2.filter2D(M, ddepth=-1, kernel=ky)
    return Ix, Iy

# load the image
image = cv2.imread('./2023_11_10_Class_exercises/images/3.jpg')
# convert to gray scale image
Iog = image.copy()
I = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# define the size of the window
size = 3
u = size
v = size

# constants
k = 0.05
sigma = 0.5
threshold = 0.005

# create w matrix
w = np.zeros((u,v))
for i in range(u):
    for j in range(v):
        w[i][j] = np.exp(-(i+j)**2/(2*sigma))

# Gx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
Gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

Gy = Gx.T

H = get_harris_corners(I, k, Gx, Gy, w)

# define the radius of the circle based on the size of the image
radius = int(I.shape[0]*0.005)

# calculate the harris matrix
for i in range(1, I.shape[0]):
    for j in range(1, I.shape[1]):
        # mark the coreners 
        if H[i][j] > threshold:
            cv2.circle(Iog, (j,i), radius, (0,255,0), 1)

# show the image with matplotlib
plt.imshow(Iog)
plt.show()