import numpy as np
import matplotlib.pyplot as plt

# define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(theta, x):
    return -2*np.exp(np.dot(-theta, x)) * x / (1 + np.exp(np.dot(-theta,x)))**2
    
# gradient based on the mean squared error
def gradient(x, y, theta):
    m = y.shape[0]
    return 1/m * np.sum((y - sigmoid(np.dot(theta, x)))*sigmoid_derivative(theta, x), axis=1)


def gradient_decend(x, y, theta, alpha, iterations):
    while iterations > 0:
        # update theta
        theta = theta + alpha * gradient(x.T, y, theta)
        iterations -= 1

    return theta

# create the data two groups of points clearely clustered more dense in one group than the other
x = np.array([[1, 1], 
              [1, 2], 
              [2, 1], 
              [2, 2], 
              [4, 2], 
              [4, 3], 
              [5, 2], 
              [5, 3],
            ])

y = np.array([0, 0, 0, 0, 1, 1, 1, 1]) #, 1, 1, 1 , 1, 1, 1, 1, 1])

# center the data
x[:,0] = x[:,0] - np.mean(x[:,0])
x[:,1] = x[:,1] - np.mean(x[:,1])

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
#find c from the mean of the points
c = 0

x_plot = np.linspace(-2, 2, 100)
y_plot = m*x_plot + c
plt.plot(x_plot, y_plot, 'r')
plt.grid()

#plot the sigmoid function
plt.figure()
plt.plot(range(len(x)), sigmoid(np.dot(theta, x.T)))

# get a ramdom sample of points 
x_test = np.random.rand(100, 2) 

# # add a translation
# x_test[:,0] = x_test[:,0]*1.2 + 1
# x_test[:,1] = x_test[:,1]*1.2 + 1
# center the data
x_test[:,0] = x_test[:,0] - np.mean(x_test[:,0])
x_test[:,1] = x_test[:,1] - np.mean(x_test[:,1])

# plot the points
plt.figure()
plt.scatter(x_test[:,0], x_test[:,1])

# apply the classifier
y_test = sigmoid(np.dot(theta, x_test.T))

# plot the points
plt.figure()
plt.scatter(x_test[:,0], x_test[:,1], c=y_test)
plt.colorbar()
# plot the line
x_plot = np.linspace(0, 1, 100)
y_plot = m*x_plot
plt.plot(x_plot, y_plot, 'r')
plt.grid()

#plot the sigmoid of the test points
plt.figure()
plt.plot(range(len(x_test)), y_test)


plt.show()
