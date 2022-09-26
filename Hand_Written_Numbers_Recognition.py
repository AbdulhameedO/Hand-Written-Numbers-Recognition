import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("D:\Road To Glory\Deep Learning\MNIST Number Recognition from scratch\Training_Data\mnist_train.csv")
data_dev_set = pd.read_csv("D:\Road To Glory\Deep Learning\MNIST Number Recognition from scratch\Training_Data\mnist_test.csv")

data = np.array(data)
data_dev_set = np.array(data_dev_set)


m , n = data.shape
#np.random.shuffle(data)

data_dev_set = data_dev_set.T
Y_dev_set = data_dev_set[0]
X_dev_set = data_dev_set[1 : n]

data_train_set = data.T
Y_train_set = data_train_set[0]
X_train_set = data_train_set[1 : n]
X_train_set = X_train_set / 255

#Defining Functions

def initialize_parameters():
    W1 = np.random.rand(10 , 784) - 0.5
    b1 = np.random.rand(10 , 1) - 0.5

    W2 = np.random.rand(10 , 10) - 0.5
    b2 = np.random.rand(10 , 1) - 0.5
    
    return W1 , b1 , W2 , b2

#Activation Function
#Change it to what suits the application
def activation_function(Z):
    return np.maximum(Z , 0)
# Do not forget to change it after changing the activation function
def deriv_activation_function(Z):
    return Z > 0

# softmax function to let the output be a percentage
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def one_hot(Y):
    hot = np.zeros((Y.size , Y.max() + 1))
    hot[np.arange(Y.size) , Y] = 1
    return hot.T 

def forward_propagation(W1 , b1 , W2 , b2 , X):
    Z1 = W1.dot(X) + b1
    A1 = activation_function(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1 , A1, Z2 , A2

def backward_propagation(Z1 , A1 , Z2 , A2 , W2 , X , Y):
    Y= one_hot(Y)
    dZ2 = A2 - Y
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2)

    dZ1 = W2.T.dot(dZ2) * deriv_activation_function(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ2)

    return dW2 , db2 , dW1 , db1

# Update parameters after backward propagation

def update_parameters(W1 , b1 , W2 , b2 , dW1 , db1 , dW2 , db2 , learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1 , b1 , W2 , b2

# Getting prediction and calculating the accuracy in the current iteration
def predictions(A2):
    return np.argmax(A2 , 0)

def accuracy(predictions , Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X , Y , iterations , learning_rate):
    W1 , b1 , W2 , b2 = initialize_parameters()
    for i in range(iterations):
        Z1 , A1, Z2 , A2 = forward_propagation(W1 , b1 , W2 , b2 , X_train_set)
        dW2 , db2 , dW1 , db1 = backward_propagation(Z1 , A1 , Z2 , A2 , W2 , X_train_set , Y_train_set)
        W1 , b1 , W2 , b2 = update_parameters(W1 , b1 , W2 , b2 , dW1 , db1 , dW2 , db2 , learning_rate)
        if i % 100== 0 or i==iterations - 1:
            print("\n\nIteration: " , i)
            print("Accuracy: " , accuracy(predictions(A2) , Y) * 100 , "%")
    
    return W1 , b1 , W2 , b2

W1 , b1 , W2 , b2 = gradient_descent(X_train_set, Y_train_set , 1000, 0.1)
