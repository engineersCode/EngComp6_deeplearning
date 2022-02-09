#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

"""The functions used in lesson 7.

Usage:
    from lesson_7_functions import *

This will make the following functions available:
    - classify
    - performance
    - neural_network_model
    - model_loss
"""
from autograd import numpy
from lesson_5_functions import logistic, classify, performance


# doing `from lesson_y_functions import *` will import these objects
__all__ = ["neural_network_model", "model_loss", "regularized_loss", "classify", "performance"]


def neural_network_model(x, params):
    """A fully-connected neural network with L=1.

    Arguments
    ---------
    x : numpy.ndarray
        The input of the model. It's shape should be (n_images, n_total_pixels).
    params : a tuple/list of four elements
        - The first element is W0, a 2D array with shape (n_total_pixels, n_z1).
        - The second elenment is b0, an 1D array with length n_z1.
        - The third element is W1, an 1D array with length n_z1.
        - The fourth element is b1, a scalar.

    Returns
    -------
    yhat : numpy.ndarray
        The predicted values obtained from the model. It's an 1D array with
        length n_images.
    """
    z1 = logistic(numpy.dot(x, params[0])+params[1])
    yhat = logistic(numpy.dot(z1, params[2])+params[3])
    return yhat


def model_loss(x, true_labels, params):
    """Calculate the predictions and the loss w.r.t. the true values.

    Arguments
    ---------
    x : numpy.ndarray
        The input of the model. The shape should be (n_images, n_total_pixels).
    true_labels : numpy.ndarray
        The true labels of the input images. Should be 1D and have length of
        n_images.
    params : a tuple/list of two elements
        - The first element is W0, a 2D array with shape (n_total_pixels, n_z1).
        - The second elenment is b0, an 1D array with length n_z1.
        - The third element is W1, an 1D array with length n_z1.
        - The fourth element is b1, a scalar.

    Returns
    -------
    loss : a scalar
        The summed loss.
    """
    pred = neural_network_model(x, params)

    n_images = x.shape[0]

    # major loss
    loss = - (
        numpy.dot(true_labels, numpy.log(pred+1e-15)) +
        numpy.dot(1.-true_labels, numpy.log(1.-pred+1e-15))
    ) / n_images

    return loss


def regularized_loss(x, true_labels, params, _lambda=1.):
    """Return the loss with regularization.

    Arguments
    ---------
    x, true_labels, params :
        Parameters for function `model_loss`.
    _lambda : float
        The weight of the regularization term. Default: 0.01

    Returns
    -------
    loss : a scalar
        The summed loss.
    """
    loss = model_loss(x, true_labels, params)
    Nw = params[0].shape[0] * params[0].shape[1] + params[2].size
    reg = ((params[0]**2).sum() + (params[2]**2).sum()) / Nw
    return loss + _lambda * reg
