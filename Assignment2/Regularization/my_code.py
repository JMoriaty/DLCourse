# import packages
import numpy as np
import matplotlib.pyplot as plt
from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
import sklearn
import sklearn.datasets
import scipy.io
from testCases import *
from regularization import *

train_X, train_Y, test_X, test_Y = load_2D_dataset()


def model(X, Y, learning_rate=0.3, num_iterations=30000, print_cost=True, lambd=0, keep_prob=1):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
    learning_rate -- learning rate of the optimization
    num_iterations -- number of iterations of the optimization loop
    print_cost -- If True, print the cost every 10000 iterations
    lambd -- regularization hyperparameter, scalar
    keep_prob - probability of keeping a neuron active during drop-out, scalar.

    Returns:
    parameters -- parameters learned by the model. They can then be used to predict.
    """

    grads = {}
    costs = []  # to keep track of the cost
    m = X.shape[1]  # number of examples
    layers_dims = [X.shape[0], 20, 3, 1]

    # Initialize parameters dictionary.
    parameters = initialize_parameters(layers_dims)

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        if keep_prob == 1:
            a3, cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)

        # Cost function
        if lambd == 0:
            cost = compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)

        # Backward propagation.
        assert (lambd == 0 or keep_prob == 1)  # it is possible to use both L2 regularization and dropout,
        # but this assignment will only explore one at a time
        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the loss every 10000 iterations
        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


if __name__ == '__main__':

    # # -------1-Non-regularized model---------
    # parameters = model(train_X, train_Y)
    # print("On the training set:")
    # predictions_train = predict(train_X, train_Y, parameters)
    # print("On the test set:")
    # predictions_test = predict(test_X, test_Y, parameters)
    # plt.title("Model without regularization")
    # axes = plt.gca()
    # axes.set_xlim([-0.75, 0.40])
    # axes.set_ylim([-0.75, 0.65])
    # plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


    # # -------2-L2 Regularization---------
    # A3, Y_assess, parameters = compute_cost_with_regularization_test_case()
    # print("cost = " + str(compute_cost_with_regularization(A3, Y_assess, parameters, lambd=0.1)))
    #
    # X_assess, Y_assess, cache = backward_propagation_with_regularization_test_case()
    #
    # grads = backward_propagation_with_regularization(X_assess, Y_assess, cache, lambd=0.7)
    # print("dW1 = \n" + str(grads["dW1"]))
    # print("dW2 = \n" + str(grads["dW2"]))
    # print("dW3 = \n" + str(grads["dW3"]))
    #
    # parameters = model(train_X, train_Y, lambd=0.7)
    # print("On the train set:")
    # predictions_train = predict(train_X, train_Y, parameters)
    # print("On the test set:")
    # predictions_test = predict(test_X, test_Y, parameters)
    #
    # plt.title("Model with L2-regularization")
    # axes = plt.gca()
    # axes.set_xlim([-0.75, 0.40])
    # axes.set_ylim([-0.75, 0.65])
    # plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


    # # -------3.1 - Forward propagation with dropout---------
    X_assess, parameters = forward_propagation_with_dropout_test_case()

    A3, cache = forward_propagation_with_dropout(X_assess, parameters, keep_prob=0.7)
    print("A3 = " + str(A3))

    X_assess, Y_assess, cache = backward_propagation_with_dropout_test_case()

    gradients = backward_propagation_with_dropout(X_assess, Y_assess, cache, keep_prob=0.8)

    print("dA1 = \n" + str(gradients["dA1"]))
    print("dA2 = \n" + str(gradients["dA2"]))

    parameters = model(train_X, train_Y, keep_prob=0.86, learning_rate=0.3)

    print("On the train set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)

    plt.title("Model with dropout")
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)