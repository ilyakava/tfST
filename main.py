"""

Example usage:
CUDA_VISIBLE_DEVICES=0 python main.py --dataset=IP --data_root=/scratch0/ilya/locDoc/data/hyperspec/datasets/ --train_test_splits=Indian_pines_gt_traintest.mat
"""


import argparse

from collections import namedtuple
import itertools
from itertools import product
import time
import os
import random

import scipy.io as sio
import h5py
import hdf5storage
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from nets import IP_net
from data import get_dataset

import pdb

layerO = namedtuple('layerO', ['strides', 'padding'])
netO = namedtuple('netO', ['model_fn', 'addl_padding'])

def hyper_run_acc(data, labels, netO, traintestfilenames=None, outfilename=None, test_egs=None, expt_name='lin'):
    """
    Args: data, image in (height, width, nbands) format
    """
    [height, width, nbands] = data.shape


    all_pixels = np.array(list(itertools.product(range(width),range(height))))
    labelled_pixels = np.array(filter(lambda (x,y): labels[y,x] != 0, all_pixels))
    flat_labels = labels.transpose().reshape(height*width)
    nlabels = len(set(flat_labels.tolist())) - 1

    ap = np.array(netO.addl_padding)
    assert np.all(ap % 2 == 0), 'Assymetric is not supported'
    net_in_shape = ap + np.array([1,1,nbands])
    x = tf.placeholder(tf.float32, shape=net_in_shape)
    feat = netO.model_fn(x)

    padded_data = np.pad(data, ((ap[0]/2,ap[0]/2),(ap[1]/2,ap[1]/2),(ap[2]/2,ap[2]/2)), 'wrap')

    print('requesting %d MB memory' % (labelled_pixels.shape[0] * feat.shape[0] * 4 / 1000000.0))
    labelled_pix_feat = np.zeros((labelled_pixels.shape[0], feat.shape[0]), dtype=np.float32)
    
    def compute_features():
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for pixel_i, pixel in enumerate(tqdm(labelled_pixels)):
            # this iterates through columns first
            [pixel_x, pixel_y] = pixel
            subimg = padded_data[pixel_y:(pixel_y+ap[0]+1), pixel_x:(pixel_x+ap[1]+1), :]
        
            feed_dict = {x: subimg}
            labelled_pix_feat[pixel_i,:] = sess.run(feat, feed_dict)
        sess.close()
        tf.reset_default_graph()

    print('computing features now')
    compute_features()

    exp_OAs = []
    for traintestfilename in traintestfilenames:
        mat_contents = None
        try:
            mat_contents = sio.loadmat(traintestfilename)
        except:
            mat_contents = hdf5storage.loadmat(traintestfilename)
        train_mask = mat_contents['train_mask'].astype(int).squeeze()
        test_mask = mat_contents['test_mask'].astype(int).squeeze()
        # resize train/test masks to labelled pixels
        # we have to index X differently than Y since only labelled feat are computed
        train_mask_skip_unlabelled = train_mask[flat_labels!=0]
        test_mask_skip_unlabelled = test_mask[flat_labels!=0]

        # get training set
        trainY = flat_labels[train_mask==1]
        trainX = labelled_pix_feat[train_mask_skip_unlabelled==1,:]


        nextoutfilename = traintestfilename+'_pyFST3D_'+expt_name+'_expt.mat'

        print('starting training')
        start = time.time()
        clf = SVC(kernel='linear')
        clf.fit(trainX, trainY)
        end = time.time()
        print(end - start)

        # now test
        test_chunk_size = 1000
        testY = flat_labels[test_mask==1]

        # we want to shuffle the feat and labels in the same order
        # and be able to unshuffle the pred_labels afterwards
        order = range(testY.shape[0]); random.shuffle(order)
        # shuffle idxs into labelled feat
        labelled_pix_feat_idxs = np.array(range(labelled_pix_feat.shape[0]))
        test_labelled_pix_feat_idxs = labelled_pix_feat_idxs[test_mask_skip_unlabelled==1]
        shuff_test_labelled_pix_feat_idxs = test_labelled_pix_feat_idxs[order]
        # and shuffle test labels
        shuff_test_labels = testY[order]

        shuff_test_pred_pix = np.zeros(testY.shape)

        C = np.zeros((nlabels,nlabels))
        print('testing now')
        mat_outdata = {}
        mat_outdata[u'metrics'] = {}
        test_limit_egs = len(testY)
        if test_egs:
            test_limit_egs = test_egs
        for i in tqdm(range(0,test_limit_egs,test_chunk_size)):
            # populate test X
            this_feat_idxs = shuff_test_labelled_pix_feat_idxs[i:i+test_chunk_size]
            this_labs = shuff_test_labels[i:i+test_chunk_size].tolist()

            p_label = clf.predict(labelled_pix_feat[this_feat_idxs]);
            shuff_test_pred_pix[i:i+test_chunk_size] = p_label
            C += confusion_matrix(this_labs, p_label, labels=list(range(1,nlabels+1)))

            mat_outdata[u'metrics'][u'CM'] = C
            hdf5storage.write(mat_outdata, filename=nextoutfilename, matlab_compatible=True)

        exp_OAs.append(100*np.diagonal(C).sum() / C.sum())
        mat_outdata[u'true_image'] = flat_labels.reshape((width, height)).transpose()
        # unshuffle predictions
        Yhat = np.zeros(testY.shape)
        for i, j in enumerate(order):
            Yhat[j] = shuff_test_pred_pix[i]
        # reshape Yhat to an image, and save for later comparison
        pred_image = np.zeros(flat_labels.shape)
        pred_image[test_mask==1] = Yhat
        mat_outdata[u'pred_image'] = pred_image.reshape((width, height)).transpose()
        hdf5storage.write(mat_outdata, filename=nextoutfilename, matlab_compatible=True)

    print('ACC ACHIEVED ({}): {:.4f}'.format(expt_name, np.array(exp_OAs).mean()))

def main(opt):
    data, labels = get_dataset(opt)

    traintestfilenames = opt.train_test_splits.split(',')

    hyper_run_acc(data, labels, IP_net(), traintestfilenames)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='IP | Bots | Pavia')
    parser.add_argument('--data_root', required=True, help='path to dataset')
    parser.add_argument('--train_test_splits', default='', help='Comma setparated list')

    opt = parser.parse_args()

    main(opt)


    