import numpy as np
import matplotlib.pyplot as plt

# define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return 2*np.exp(-x) * x / (1 + np.exp(-x))**2
    

def loss_function(x, y, theta):
    m = y.shape[0]
    return np.sum(y - sigmoid(np.dot(theta, x))**2)*1/m

# gradient based on the mean squared error
def gradient(x, y, theta):
    m = y.shape[0]
    return 1/m * np.sum(y - sigmoid(np.dot(theta, x)))*sigmoid_derivative(np.dot(theta, x))

def gradient_decend(x, y, theta, alpha, iterations):
    # initialize theta 
    theta = np.zeros(x.shape[1])

    while iterations > 0:
        # update theta
        theta = theta + alpha * gradient(x, y, theta)
        iterations -= 1

    return theta

# create the data 
x = np.array([[1,2],[3,4],[5,6],[7,8],[9,10]])
y = np.array([0,0,1,1,1])

# plot the data
plt.scatter(x[:,1], y)
plt.show()
# initialize theta
theta = np.zeros(x.shape[1])

# define the learning rate
alpha = 0.01
# define the number of iterations
iterations = 100

# compute the gradient decend
# theta = gradient_decend(x.T, y, theta, alpha, iterations)
        
# # plot the data
plt.figure()
plt.scatter(x[:,1], y)
plt.plot(x[:,1], sigmoid(np.dot(x, theta)))
plt.show()


