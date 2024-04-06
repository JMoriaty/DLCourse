import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from PIL import Image
from dnn_app_utils_v2 import *

# plt.rcParams['figure.figsize'] = (5.0, 4.0)
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()


def showdata():
    index = 10
    plt.imshow(train_x_orig[index])
    print("y = " + str(train_y[0, index]) + ". It is a " + classes[train_y[0,index]].decode("utf-8") + " picture.")

    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]

    print("Number of training examples: " + str(m_train))
    print("Number of testing examples:" + str(m_test))
    print("Each image is of size: (" + str(num_px) + "," + str(num_px) + ",3)")
    print("train_x_orig's shape " + str(train_x_orig.shape))
    print("train_y's shape " + str(train_y))
    print("test_x_orig's shape " + str(test_x_orig.shape))
    print("test_y's shape " + str(test_y.shape))


def StandardizeImage():
    # Reshape the training and test examples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    print("train_x's shape " + str(train_x.shape))
    print("test_x's shape " + str(test_x.shape))


def initialize_parameters_deep(layer_dims):
    parameters = {}
    L = len(layer_dims)

    for i in range(1, L):
        parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[1-i]) * 0.01
        parameters['b' + str(i)] = np.random.randn(layer_dims[1], 1)
        assert(parameters['W' + str(i)].shape == (layer_dims[1], layer_dims[i-1]))
        assert(parameters['b' + str(i)].shape == (layer_dims[1], 1))

    return parameters


def activation_function(A_prev, W, b, activation):
    if activation == 'sigmoid':
        Z = np.dot(W, A_prev) + b
        linear_cache = (A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == 'relu':
        Z = np.dot(W, A_prev) + b
        linear_cache = (A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2

    for i in range(L):
        A_prev = A
        A, caches = activation_function(A_prev, parameters['W' + str(i)], parameters['b' + str(i)], activation='relu')
        caches.append(caches)

    AL, caches = activation_function(A, parameters['W' + str(L)], parameters['b' + str(L)], activation='relu')
    caches.append(caches)

    return AL, caches


def computer_cost(AL, Y):
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(Y * np.log(AL) + (1 - AL) * np.log(1 - AL))
    cost = np.squeeze(cost, axis=0)
    assert(cost.shape == ())

    return cost


def L_model_backward(AL, Y, caches):


def update_parameters(parameters, grads,learning_rate):


if __name__ == '__main__':
    showdata()
    StandardizeImage()