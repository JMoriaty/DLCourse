import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from opt_utils import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from testCases import *

from optimization import *


def model(X, Y, layers_dims, optimizer, learning_rate=0.0007, mini_batch_size=64, beta=0.9,
          beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000, print_cost=True):
    """
    3-layer neural network model which can be run in different optimizer modes.

    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    layers_dims -- python list, containing the size of each layer
    learning_rate -- the learning rate, scalar.
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_cost -- True to print the cost every 1000 epochs

    Returns:
    parameters -- python dictionary containing your updated parameters
    """

    L = len(layers_dims)  # number of layers in the neural networks
    costs = []  # to keep track of the cost
    t = 0  # initializing the counter required for Adam update
    seed = 10  # For grading purposes, so that your "random" minibatches are the same as ours
    m = X.shape[1]  # number of training examples

    # Initialize parameters
    parameters = initialize_parameters(layers_dims)

    # Initialize the optimizer
    if optimizer == "gd":
        pass  # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    # Optimization loop
    for i in range(num_epochs):

        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            a3, caches = forward_propagation(minibatch_X, parameters)

            # Compute cost and add to the cost total
            cost_total += compute_cost(a3, minibatch_Y)

            # Backward propagation
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)

            # Update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1  # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2, epsilon)
        cost_avg = cost_total / m

        # Print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print("Cost after epoch %i: %f" % (i, cost_avg))
        if print_cost and i % 100 == 0:
            costs.append(cost_avg)

    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters


if __name__ == '__main__':
    # # ------------------Gradient Descent----------------------
    # parameters, grads, learning_rate = update_parameters_with_gd_test_case()
    #
    # parameters = update_parameters_with_gd(parameters, grads, learning_rate)
    # print("W1 =\n" + str(parameters["W1"]))
    # print("b1 =\n" + str(parameters["b1"]))
    # print("W2 =\n" + str(parameters["W2"]))
    # print("b2 =\n" + str(parameters["b2"]))


    # # ------------------2 - Mini-Batch Gradient descent----------------------
    # X_assess, Y_assess, mini_batch_size = random_mini_batches_test_case()
    # mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)
    #
    # print("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
    # print("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
    # print("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
    # print("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
    # print("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape))
    # print("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
    # print("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))


    # # ------------------3 - Momentum----------------------
    # parameters = initialize_velocity_test_case()
    #
    # v = initialize_velocity(parameters)
    # print("v[\"dW1\"] =\n" + str(v["dW1"]))
    # print("v[\"db1\"] =\n" + str(v["db1"]))
    # print("v[\"dW2\"] =\n" + str(v["dW2"]))
    # print("v[\"db2\"] =\n" + str(v["db2"]))
    #
    # parameters, grads, v = update_parameters_with_momentum_test_case()
    #
    # parameters, v = update_parameters_with_momentum(parameters, grads, v, beta=0.9, learning_rate=0.01)
    # print("W1 = \n" + str(parameters["W1"]))
    # print("b1 = \n" + str(parameters["b1"]))
    # print("W2 = \n" + str(parameters["W2"]))
    # print("b2 = \n" + str(parameters["b2"]))
    # print("v[\"dW1\"] = \n" + str(v["dW1"]))
    # print("v[\"db1\"] = \n" + str(v["db1"]))
    # print("v[\"dW2\"] = \n" + str(v["dW2"]))
    # print("v[\"db2\"] = v" + str(v["db2"]))


    # # ------------------4 - Adam----------------------
    # parameters = initialize_adam_test_case()
    #
    # v, s = initialize_adam(parameters)
    # print("v[\"dW1\"] = \n" + str(v["dW1"]))
    # print("v[\"db1\"] = \n" + str(v["db1"]))
    # print("v[\"dW2\"] = \n" + str(v["dW2"]))
    # print("v[\"db2\"] = \n" + str(v["db2"]))
    # print("s[\"dW1\"] = \n" + str(s["dW1"]))
    # print("s[\"db1\"] = \n" + str(s["db1"]))
    # print("s[\"dW2\"] = \n" + str(s["dW2"]))
    # print("s[\"db2\"] = \n" + str(s["db2"]))
    #
    # parameters, grads, v, s = update_parameters_with_adam_test_case()
    # parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t=2)
    #
    # print("W1 = \n" + str(parameters["W1"]))
    # print("b1 = \n" + str(parameters["b1"]))
    # print("W2 = \n" + str(parameters["W2"]))
    # print("b2 = \n" + str(parameters["b2"]))
    # print("v[\"dW1\"] = \n" + str(v["dW1"]))
    # print("v[\"db1\"] = \n" + str(v["db1"]))
    # print("v[\"dW2\"] = \n" + str(v["dW2"]))
    # print("v[\"db2\"] = \n" + str(v["db2"]))
    # print("s[\"dW1\"] = \n" + str(s["dW1"]))
    # print("s[\"db1\"] = \n" + str(s["db1"]))
    # print("s[\"dW2\"] = \n" + str(s["dW2"]))
    # print("s[\"db2\"] = \n" + str(s["db2"]))


    # ------------------5 - Model with different optimization algorithms----------------------
    train_X, train_Y = load_dataset()

    # train 3-layer model
    # optimizer = [gd, momentum, adam]
    layers_dims = [train_X.shape[0], 5, 2, 1]
    parameters = model(train_X, train_Y, layers_dims, optimizer="adam")

    # Predict
    predictions = predict(train_X, train_Y, parameters)

    # Plot decision boundary
    plt.title("Model with Adam optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 2.5])
    axes.set_ylim([-1, 1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

