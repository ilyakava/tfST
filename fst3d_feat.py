"""FST-svm
"""

from collections import namedtuple
import itertools
from itertools import product
import time
import os
import random

import h5py
import hdf5storage
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import scipy.io as sio
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

import windows as win

import pdb


def scat3d(x, win_params, layer_params):
    """
    Args:
        x is input with dim (batch, depth, height, width, channels)
        win_params.filters is complex with dim (depth, height, width, channels)
    """
    real1 = tf.layers.conv3d(
        x,
        win_params.nfilt,
        win_params.kernel_size,
        strides=layer_params.strides,
        padding=layer_params.padding,
        dilation_rate=(1,1,1),
        activation=None,
        use_bias=False,
        kernel_initializer=tf.constant_initializer(win_params.filters.real, dtype=tf.float32),
        trainable=False,
        name=None
    )

    imag1 = tf.layers.conv3d(
        x,
        win_params.nfilt,
        win_params.kernel_size,
        strides=layer_params.strides,
        padding=layer_params.padding,
        dilation_rate=(1,1,1),
        activation=None,
        use_bias=False,
        kernel_initializer=tf.constant_initializer(win_params.filters.imag, dtype=tf.float32),
        trainable=False,
        name=None
    )

    return tf.abs(tf.complex(real1, imag1))

def kernel_padding(kernel_size):
    def int_padding(n):
        if n % 2 == 0:
            raise('not implemented even padding')
        else:
            return int((n - 1) / 2)
    return [int_padding(m) for m in kernel_size]

def hyper3d_net(x, reuse=tf.AUTO_REUSE, psis=None, phi=None, layer_params=None):
    """Computes features for a specific pixel.

    Args:
        x: image in (height, width, bands) format
        psis: array of winO struct, filters are in (bands, height, width) format!
        phi: winO struct, filters are in (bands, height, width) format!
    Output:
        center pixel feature vector
    """
    assert len(layer_params) == 3, 'this network is 2 layers only'
    assert len(psis) == 2, 'this network is 2 layers only'

    
    with tf.variable_scope('Hyper3DNet', reuse=reuse):
        x = tf.transpose(x, [2, 0, 1])

        x = tf.expand_dims(x, 0)
        x = tf.expand_dims(x, -1)
        # x is (1, bands, h, w, 1)

        U1 = scat3d(x, psis[0], layer_params[0])
        # U1 is (1, bands, h, w, lambda1)


        # swap channels with batch
        U1 = tf.transpose(U1, [4, 1, 2, 3, 0])
        # U1 is (lambda1, bands, h, w, 1)
        
        U2s = []
        # only procede with increasing frequency paths
        for res_i, used_params in enumerate(psis[0].filter_params):
            increasing_psi = win.fst3d_psi_factory(psis[1].kernel_size, used_params)
            if increasing_psi.nfilt > 0:
                U2s.append(scat3d(U1[res_i:(res_i+1),:,:,:,:], increasing_psi, layer_params[1]))

        U2 = tf.concat(U2s, 4)
        # swap channels with batch
        U2 = tf.transpose(U2, [4, 1, 2, 3, 0])

        # convolve with phis
        S2 = scat3d(U2, phi, layer_params[2])

        [p1h, p1w, p1b] = kernel_padding(psis[1].kernel_size)
        [p2h, p2w, p2b] = kernel_padding(psis[0].kernel_size)
        p2h += p1h; p2w += p1w; p2b += p1b;

        S1 = scat3d(U1[:,(p1h):-(p1h), (p1w):-(p1w), (p1b):-(p1b), :], phi, layer_params[2])
        
        S0 = scat3d(x[:,(p2h):-(p2h), (p2w):-(p2w), (p2b):-(p2b), :], phi, layer_params[2])

        # flatten everything
        S2 = tf.reshape(S2, [S2.shape[0] * S2.shape[1]]) # enforces last 3 dimensions being 1
        S1 = tf.reshape(S1, [S1.shape[0] * S1.shape[1]]) # enforces last 3 dimensions being 1
        S0 = tf.reshape(S0, [S0.shape[1]]) # enforces all but dim1 being 1

    return tf.concat([S0,S1,S2], 0)

