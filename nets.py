"""
"""

import tensorflow as tf
from collections import namedtuple
from fst3d_feat import hyper3d_net
import windows as win

layerO = namedtuple('layerO', ['strides', 'padding'])
netO = namedtuple('netO', ['model_fn', 'addl_padding'])

def IP_net(reuse=tf.AUTO_REUSE):
    """Fully described network.

    This method is basically "data as code"
        
    Returns:
        struct with fields:
        addl_padding: amount of padding needed for an input to model_fn
        model_fn: function that takes:
            
    """
    psi = win.fst3d_psi_factory([3,9,9])
    phi = win.fst3d_phi_window_3D([3,9,9])
    layer_params = layerO((3,1,1), 'valid')

    def model_fn(x):
        """
        Args:
            x: tf.placeholder in (height, width, nbands) format, shape must be (25,25,nbands+6)
        """
        return hyper3d_net(x, reuse=reuse, psis=[psi,psi],
            phi=phi, layer_params=[layer_params, layer_params, layer_params])

    return netO(model_fn, (24,24,6))

def Bots_net(reuse=tf.AUTO_REUSE):
    s = 11
    psi1 = win.fst3d_psi_factory([3,s,s])
    psi2 = win.fst3d_psi_factory([8,s,s])
    phi = win.fst3d_phi_window_3D([8,s,s])
    lp1 = layerO((1,1,1), 'valid')
    lp2 = layerO((8,1,1), 'valid')
    lp3 = layerO((8,1,1), 'valid')

    def model_fn(x):
        """
        Args:
            x: tf.placeholder in (height, width, nbands) format, shape must be (25,25,nbands+6)
        """
        return hyper3d_net(x, reuse=reuse, psis=[psi1,psi2],
            phi=phi, layer_params=[lp1, lp2, lp3])

    return netO(model_fn, ((s-1)*3,(s-1)*3,0))

def Pavia_net(reuse=tf.AUTO_REUSE):
    psi = win.fst3d_psi_factory([7,7,7])
    phi = win.fst3d_phi_window_3D([7,7,7])
    layer_params = layerO((3,1,1), 'valid')

    def model_fn(x):
        """
        Args:
            x: tf.placeholder in (height, width, nbands) format, shape must be (25,25,nbands+6)
        """
        return hyper3d_net(x, reuse=reuse, psis=[psi,psi],
            phi=phi, layer_params=[layer_params, layer_params, layer_params])
    return netO(model_fn, (18,18,18))

def PaviaR_net(reuse=tf.AUTO_REUSE):
    psi = win.fst3d_psi_factory([3,9,9])
    phi = win.fst3d_phi_window_3D([3,9,9])
    layer_params = layerO((3,1,1), 'valid')

    def model_fn(x):
        """
        Args:
            x: tf.placeholder in (height, width, nbands) format, shape must be (25,25,nbands+6)
        """
        return hyper3d_net(x, reuse=reuse, psis=[psi,psi],
            phi=phi, layer_params=[layer_params, layer_params, layer_params])
    return netO(model_fn, (24,24,6))
