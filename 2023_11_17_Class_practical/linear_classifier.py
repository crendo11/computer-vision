import numpy as np
import matplotlib.pyplot as plt

# define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(theta, x):
    return 2*np.exp(np.dot(-theta, x)) * x / (1 + np.exp(np.dot(-theta,x)))**2
    

def loss_function(x, y, theta):
    m = y.shape[0]
    return np.sum(y - sigmoid(np.dot(theta, x))**2)*1/m

# gradient based on the mean squared error
def gradient(x, y, theta):
    m = y.shape[0]
    return 1/m * np.sum(y - sigmoid(np.dot(theta, x))*sigmoid_derivative(theta, x))

def gradient_decend(x, y, theta, alpha, iterations):
    while iterations > 0:
        # update theta
        theta = theta + alpha * gradient(x.T, y, theta)
        iterations -= 1

    return theta

# create the data two groups of points clearely clustered
x = np.array([[1, 1], 
              [1, 2], 
              [2, 1], 
              [2, 2], 
              [4+4, 4], 
              [4+4, 5], 
              [5+4, 4], 
              [5+4, 5]])

y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# initialize theta
theta = np.zeros(x.shape[1])     # the shape of theta is the number of columns of x

# define the learning rate
alpha = 0.01
# define the number of iterations
iterations = 100

# compute the gradient decend
theta = gradient_decend(x, y, theta, alpha, iterations)
print(theta)
        
# # plot the data
plt.figure()
plt.scatter(x[:,1], y)

# also plot the points
plt.figure()
plt.scatter(x[:,0], x[:,1])

# plot the line
m = -theta[0]/theta[1]
c = 0
x_plot = np.linspace(0, 6, 100)
y_plot = m*x_plot + c
plt.plot(x_plot, y_plot, 'r')

#plot the sigmoid function
plt.figure()
plt.plot(x[:,1], sigmoid(np.dot(theta, x.T)))


plt.show()
