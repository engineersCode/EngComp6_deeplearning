#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

"""Load metal-casting datasets.

Usage:
    frpm load_casting_data import *

Then the following variables will be available:
    - res: image resolution in both x and y
    - n_ok_total: total number of images for normal parts
    - n_ok_train: number of images for normal parts in the training dataset
    - n_ok_val: number of images for normal parts in the validation dataset
    - n_ok_test: number of images for normal parts in the test dataset
    - n_def_total: total number of images for defective parts
    - n_def_train: number of images for defective parts in the training dataset
    - n_def_val: number of images for defective parts in the validation dataset
    - n_def_test: number of images for defective parts in the test dataset
    - images_train: training dataset, 2D array with shape (n_ok_train+n_def_train, res*res)
    - images_val: validation dataset, 2D array with shape (n_ok_val+n_def_val, res*res)
    - images_test: test dataset, 2D array with shape (n_ok_test+n_def_test, res*res)
    - labels_train: labels for the training dataset, 1D array with length n_ok_train+n_def_train
    - labels_val: labels for the validation dataset, 1D array with length n_ok_train+n_def_train
    - labels_test: labels for the test dataset, 1D array with length n_ok_train+n_def_train
    - mu: the mean values of training data
    - sigma: the standard deviations of training data
"""
import numpy
import pathlib

# define what will be imported when doing `from load_casting_data import *`
__all__ = [
    "res", "n_ok_total", "n_ok_train", "n_ok_val", "n_ok_test", "n_def_total", "n_def_train",
    "n_def_val", "n_def_test", "images_train", "images_val", "images_test", "labels_train",
    "labels_val", "labels_test", "mu", "sigma"
]

# path to the repository folder
root = pathlib.Path(__file__).parents[1]

# read in images and labels
with numpy.load(root.joinpath("data", "casting_images.npz")) as data:
    ok_images = data["ok_images"]
    def_images = data["def_images"]

# get the number of images and image resolution
n_ok_total = ok_images.shape[0]
n_def_total = def_images.shape[0]
res = int(numpy.sqrt(def_images.shape[1]))

# numbers of images for validation, test, and training data
n_ok_val = int(n_ok_total * 0.2)
n_def_val = int(n_def_total * 0.2)
n_ok_test = int(n_ok_total * 0.2)
n_def_test = int(n_def_total * 0.2)
n_ok_train = n_ok_total - n_ok_val - n_ok_test
n_def_train = n_def_total - n_def_val - n_def_test

# split dataset
ok_images = numpy.split(ok_images, [n_ok_val, n_ok_val+n_ok_test], 0)
def_images = numpy.split(def_images, [n_def_val, n_def_val+n_def_test], 0)

# combine images of defective and normal parts
images_val = numpy.concatenate([ok_images[0], def_images[0]], 0)
images_test = numpy.concatenate([ok_images[1], def_images[1]], 0)
images_train = numpy.concatenate([ok_images[2], def_images[2]], 0)

# calculate mu and sigma and the normalize datasets
mu = numpy.mean(images_train, axis=0)
sigma = numpy.std(images_train, axis=0)

# normalize the training, validation, and test datasets using mu and sigma
images_train = (images_train - mu) / sigma
images_val = (images_val - mu) / sigma
images_test = (images_test - mu) / sigma

# labels for training, validation, and test data
labels_train = numpy.zeros(n_ok_train+n_def_train)
labels_train[n_ok_train:] = 1.
labels_val = numpy.zeros(n_ok_val+n_def_val)
labels_val[n_ok_val:] = 1.
labels_test = numpy.zeros(n_ok_test+n_def_test)
labels_test[n_ok_test:] = 1.
