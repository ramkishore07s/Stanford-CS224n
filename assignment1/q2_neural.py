#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(X, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    the backward propagation for the gradients for all parameters.

    Notice the gradients computed here are different from the gradients in
    the assignment sheet: they are w.r.t. weights, not inputs.

    Arguments:
    X -- M x Dx matrix, where each row is a training example x.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Note: compute cost based on `sum` not `mean`.
    ### YOUR CODE HERE: forward propagation

    hidden = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden)

    output = np.dot(hidden_output, W2) + b2
    y_cap = softmax(output)

    cost = np.sum(-labels * np.log(y_cap))
    
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation

    # y_diff = np.sum(np.array([y_cap - labels]), axis = 1)
    # sig_grad = np.sum(np.array([sigmoid_grad(hidden_output)]), axis = 1)

    # cum_x = np.sum(np.array([X]), axis=1)
    # dce_dz1 = np.dot(y_diff, W2.T) * sig_grad

    # gradW1 = (cum_x.T * dce_dz1)
    # gradb1 = dce_dz1

    # cum_h = np.sum(np.array([hidden_output]), axis = 1)

    # gradW2 = (cum_h.T * y_diff)
    # gradb2 = y_diff

    y_diff = y_cap - labels
    sig_grad = sigmoid_grad(hidden_output)

    gradb2 = np.sum(y_diff, axis=0)
    gradW2 = np.dot(hidden_output.T, y_diff)

    dce_da1 = sig_grad * np.dot(y_diff, W2.T)

    gradb1 = np.sum(dce_da1, axis=0)
    gradW1 = np.dot(X.T, dce_da1)
    

    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
