"""activation_np.py
This file provides activation functions for the NN 
Author: Phuong Hoang
"""

import numpy as np


def sigmoid(x):
    """sigmoid
    TODO:
    Sigmoid function. Output = 1 / (1 + exp(-x)).
    :param x: input
    """
    # [TODO 1.1]
    output = 1 / (1 + np.exp(-x))
    return output


def sigmoid_grad(a):
    """sigmoid_grad
    TODO:
    Compute gradient of sigmoid with respect to input. g'(x) = g(x)*(1-g(x))
    :param a: output of the sigmoid function
    """
    # [TODO 1.1]
    grad = a * (1 - a)
    return grad


def reLU(x):
    """reLU
    TODO:
    Rectified linear unit function. Output = max(0,x).
    :param x: input
    """
    # [TODO 1.1]
    output = np.maximum(0, x)
    return output


def reLU_grad(a):
    """reLU_grad
    TODO:
    Compute gradient of ReLU with respect to input
    :param x: output of ReLU
    """
    # [TODO 1.1]
    return np.where(a > 0, 1, 0)


def tanh(x):
    """tanh
    TODO:
    Tanh function.
    :param x: input
    """
    # [TODO 1.1]
    output = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return output


def tanh_grad(a):
    """tanh_grad
    TODO:
    Compute gradient for tanh w.r.t input
    :param a: output of tanh
    """
    # [TODO 1.1]
    grad = 1 - a ** 2
    return grad


def softmax(x):
    """softmax
    TODO:
    Softmax function.
    :param x: input
    """

    exp_x = np.exp(x)
    output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return output


def softmax_minus_max(x):
    """softmax_minus_max
    TODO:
    Stable softmax function.
    :param x: input
    """
    minus_max_x = x - np.max(x)
    output = softmax(minus_max_x)
    return output
