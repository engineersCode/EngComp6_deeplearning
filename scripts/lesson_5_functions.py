#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
"""Functions from lesson 5.

Usage:
    from lesson_5_functions import *

This will make the following functions available:
    - logistic
    - classify
    - performance
"""
from autograd import numpy


# doing `from lesson_5_functions import *` will import these objects
__all__ = ["logistic", "classify", "performance"]


def logistic(x):
    """Logistic/sigmoid function.

    Arguments
    ---------
    x : numpy.ndarray
        The input to the logistic function.

    Returns
    -------
    numpy.ndarray
        The output.

    Notes
    -----
    The function does not restrict the shape of the input array. The output
    has the same shape as the input.
    """
    x = numpy.clip(x, -300., 300.)
    return 1. / (1. + numpy.exp(-x))


def classify(x, params, model):
    """Use a logistic model to label data with 0 or/and 1.

    Arguments
    ---------
    x : numpy.ndarray
        The input of the model. The shape should be (n_images, n_total_pixels).
    params : a tuple/list of two elements
        The first element is a 2D array with shape (n_total_pixels, 1). The
        second elenment is a scalar.
    model : a callable object
        The model that takes in `x` and `params` and then returns the probabilities.

    Returns
    -------
    labels : numpy.ndarray
        The shape of the label is the same with `probability`.

    Notes
    -----
    This function only works with multiple images, i.e., x has a shape of
    (n_images, n_total_pixels).
    """
    probabilities = model(x, params)
    labels = (probabilities >= 0.5).astype(float)
    return labels


def performance(predictions, answers, beta=1.0):
    """Calculate precision, recall, and F-score.

    Arguments
    ---------
    predictions : numpy.ndarray of integers
        The predicted labels.
    answers : numpy.ndarray of integers
        The true labels.
    beta : float
        A coefficient representing the weight of recall.

    Returns
    -------
    precision, recall, score : float
        Precision, recall, and F-score, respectively.
    """
    true_idx = (answers == 1)  # the location where the answers are 1
    false_idx = (answers == 0)  # the location where the answers are 0

    # true positive: answers are 1 and predictions are also 1
    n_tp = numpy.count_nonzero(predictions[true_idx] == 1)

    # false positive: answers are 0 but predictions are 1
    n_fp = numpy.count_nonzero(predictions[false_idx] == 1)

    # true negative: answers are 0 and predictions are also 0
    n_tn = numpy.count_nonzero(predictions[false_idx] == 0)

    # false negative: answers are 1 but predictions are 0
    n_fn = numpy.count_nonzero(predictions[true_idx] == 0)

    # precision, recall, and f-score
    precision = n_tp / (n_tp + n_fp)
    recall = n_tp / (n_tp + n_fn)
    score = (
        (1.0 + beta**2) * precision * recall /
        (beta**2 * precision + recall)
    )

    return precision, recall, score
