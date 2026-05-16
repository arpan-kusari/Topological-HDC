import numpy as np
from topo_hdc.features.extractor import FeatureBatch
from topo_hdc.hdc.encoders import Encoders
from topo_hdc.utils.Helper_functions import *

def iter_index_batches(N, batch_size, shuffle=False, rng=None):
    idx = np.arange(N)
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        rng.shuffle(idx)
    for i in range(0, N, batch_size):
        yield idx[i:i+batch_size]
        
import torch

def make_hv_generators(features, encoders, roles, batch_size):
    """
    features: FeatureBatch for train/val/test
    encoders: object or dict containing enc_h, enc_z, enc_holes
    roles: object or dict containing role_hog, role_outer, role_holes
    """

    @torch.no_grad()
    def hog_batches(shuffle=False, seed=0):
        N = features.Fh.shape[0]

        for idxb in iter_index_batches(N, batch_size, shuffle=shuffle):
            Hh = encoders.enc_h.encode_batch(
                features.Fh[idxb],
                batch_size=batch_size,
            )
            Hh = bind(Hh, roles.role_hog)
            yield idxb, Hh

    @torch.no_grad()
    def zernike_batches(shuffle=False, seed=0):
        N = features.Fz.shape[0]

        for idxb in iter_index_batches(N, batch_size, shuffle=shuffle):
            Hz = encoders.enc_z.encode_batch(
                features.Fz[idxb],
                batch_size=batch_size,
            )
            Hz = bind(Hz, roles.role_outer)
            yield idxb, Hz

    @torch.no_grad()
    def holes_batches(shuffle=False):
        N = features.HF.shape[0]

        for idxb in iter_index_batches(N, batch_size, shuffle=shuffle):
            Hholes, no_holes = encoders.enc_holes.encode_padded(
                features.HF[idxb],
                features.M[idxb],
            )

            # Optional but consistent if using role hypervectors
            Hholes = bind(Hholes, roles.role_holes)

            yield idxb, Hholes

    return {
        "hog": hog_batches,
        "zernike": zernike_batches,
        "holes": holes_batches,
    }

