import os
import numpy as np
import scipy.io as sio

def get_dataset(opt):
    if opt.dataset == 'IP':
        mat_contents = sio.loadmat(os.path.join(opt.data_root, 'Indian_pines_corrected.mat'))
        data = mat_contents['indian_pines_corrected'].astype(np.float32)
        data /= np.max(np.abs(data))
        mat_contents = sio.loadmat(os.path.join(opt.data_root, 'Indian_pines_gt.mat'))
        labels = mat_contents['indian_pines_gt']
    else:
        raise NotImplementedError('dataset: %s' % opt.dataset)
    return data, labels

