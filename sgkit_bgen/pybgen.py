"""BGEN reader implementation (using bgen_reader)"""
from pathlib import Path
from typing import Any, Union

import dask.array as da
import numpy as np
import xarray as xr
from pybgen import PyBGEN
from pybgen.parallel import ParallelPyBGEN
from xarray import Dataset

from sgkit import create_genotype_dosage_dataset
from sgkit.typing import ArrayLike
from sgkit.utils import encode_array

PathType = Union[str, Path]


def _array_name(f, path):
    return f.__qualname__ + ':' + str(path)


class BgenReader(object):

    def __init__(self, path, dtype=np.float32):
        self.path = str(path) # pybgen needs string paths

        # Use ParallelPyBGEN only to get all the variant seek positions from the BGEN index.
        # No parallel IO happens here.
        with ParallelPyBGEN(self.path) as bgen:
            bgen._get_all_seeks()
            self._seeks = bgen._seeks
            n_variants = bgen.nb_variants
            n_samples = bgen.nb_samples

            self.shape = (n_variants, n_samples)
            self.dtype = dtype
            self.ndim = 2

            self.sample_id = bgen.samples
            # This may need chunking for large numbers of variants
            variants = list(bgen.iter_variant_info())
            self.variant_id = [v.name for v in variants]
            self.contig = [v.chrom for v in variants]
            self.pos = [v.pos for v in variants]
            self.a1 = [v.a1 for v in variants]
            self.a2 = [v.a2 for v in variants]

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            raise IndexError(f'Indexer must be tuple (received {type(idx)})')
        if len(idx) != self.ndim:
            raise IndexError(f'Indexer must be two-item tuple (received {len(idx)} slices)')

        # Restrict to seeks for this chunk
        seeks_for_chunk = self._seeks[idx[0]]
        if len(seeks_for_chunk) == 0:
            return np.empty((0, 0), dtype=self.dtype)
        with PyBGEN(self.path, probs_only=True) as bgen:
            shape = (len(seeks_for_chunk), idx[1].stop - idx[1].start)
            res = np.zeros(shape, dtype=self.dtype)
            for i, (_, probs) in enumerate(bgen._iter_seeks(seeks_for_chunk)):
                res[i] = _to_dosage(probs[idx[1]])
            return res

        
def _to_dosage(probs: ArrayLike):
    """Calculate the dosage from genotype likelihoods (probabilities)"""
    assert len(probs.shape) == 2 and probs.shape[1] == 3
    return 2 * probs[:, -1] + probs[:, 1]

def read_bgen(path: PathType, chunks='auto', lock=False):

    bgen_reader = BgenReader(path)

    vars = {
        #"sample_id": xr.DataArray(np.array(bgen_reader.sample_id), dims=["sample"]),
        "variant_id": xr.DataArray(np.array(bgen_reader.variant_id), dims=["variants"]),
        "contig": xr.DataArray(np.array(bgen_reader.contig), dims=["variants"]),
        "pos": xr.DataArray(np.array(bgen_reader.pos), dims=["variants"]),
        "a1": xr.DataArray(np.array(bgen_reader.a1), dims=["variants"]),
        "a2": xr.DataArray(np.array(bgen_reader.a2), dims=["variants"]),
    }

    call_dosage = da.from_array(
        bgen_reader,
        chunks=chunks,
        lock=lock,
        asarray=False,
        name=_array_name(read_bgen, path))

    # pylint: disable=no-member
    ds = xr.Dataset(data_vars=dict(call_dosage=xr.DataArray(
        call_dosage, dims=('variants', 'samples')
    )))
    ds = ds.assign(vars)
    return ds