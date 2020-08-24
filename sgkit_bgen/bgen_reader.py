"""BGEN reader implementation (using bgen_reader)"""
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import dask
import dask.array as da
import numpy as np
import xarray as xr
from bgen_reader._bgen_file import bgen_file
from bgen_reader._bgen_metafile import bgen_metafile
from bgen_reader._metafile import create_metafile
from bgen_reader._reader import infer_metafile_filepath
from bgen_reader._samples import generate_samples, read_samples_file
from dask import Array
from xarray import Dataset

from sgkit import create_genotype_dosage_dataset
from sgkit.typing import ArrayLike
from sgkit.utils import encode_array

PathType = Union[str, Path]

STRING_VARS = ["variant_id", "variant_allele", "sample_id"]


def _to_dict(df, dtype=None):
    return {
        c: df[c].to_dask_array(lengths=True).astype(dtype[c] if dtype else df[c].dtype)
        for c in df
    }


VARIANT_FIELDS = [
    ("id", str, "U"),
    ("rsid", str, "U"),
    ("chrom", str, "U"),
    ("pos", str, "int32"),
    ("nalleles", str, "int8"),
    ("allele_ids", str, "U"),
    ("vaddr", str, "int64"),
]
VARIANT_DF_DTYPE = dict([(f[0], f[1]) for f in VARIANT_FIELDS])
VARIANT_ARRAY_DTYPE = dict([(f[0], f[2]) for f in VARIANT_FIELDS])


class BgenReader:

    name = "bgen_reader"

    def __init__(self, path, persist=True, dtype=np.float32):
        self.path = Path(path)

        self.metafile_filepath = infer_metafile_filepath(Path(self.path))
        if not self.metafile_filepath.exists():
            create_metafile(path, self.metafile_filepath, verbose=False)

        with bgen_metafile(self.metafile_filepath) as mf:
            self.n_variants = mf.nvariants
            self.npartitions = mf.npartitions
            self.partition_size = mf.partition_size

            df = mf.create_variants()
            if persist:
                df = df.persist()
            variant_arrs = _to_dict(df, dtype=VARIANT_ARRAY_DTYPE)

            self.variant_id = variant_arrs["id"]
            self.contig = variant_arrs["chrom"]
            self.pos = variant_arrs["pos"]

            def split_alleles(alleles, block_info=None):
                if block_info is None or len(block_info) == 0:
                    return alleles

                def split(allele_row):
                    alleles_list = allele_row[0].split(",")
                    assert len(alleles_list) == 2  # bi-allelic
                    return np.array(alleles_list)

                return np.apply_along_axis(split, 1, alleles[:, np.newaxis])

            variant_alleles = variant_arrs["allele_ids"].map_blocks(split_alleles)

            def max_str_len(arr: ArrayLike) -> Any:
                return arr.map_blocks(
                    lambda s: np.char.str_len(s.astype(str)), dtype=np.int8
                ).max()

            max_allele_length = max(max_str_len(variant_alleles).compute())
            self.variant_alleles = variant_alleles.astype(f"S{max_allele_length}")

        with bgen_file(self.path) as bgen:
            sample_path = self.path.with_suffix(".sample")
            if sample_path.exists():
                self.sample_id = read_samples_file(sample_path, verbose=False)
            else:
                if bgen.contain_samples:
                    self.sample_id = bgen.read_samples()
                else:
                    self.sample_id = generate_samples(bgen.nsamples)

        self.shape = (self.n_variants, len(self.sample_id), 3)
        self.dtype = dtype
        self.ndim = 3

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            raise IndexError(  # pragma: no cover
                f"Indexer must be tuple (received {type(idx)})"
            )
        if len(idx) != self.ndim:
            raise IndexError(  # pragma: no cover
                f"Indexer must be two-item tuple (received {len(idx)} slices)"
            )
        if not all(isinstance(i, slice) or isinstance(i, int) for i in idx):
            raise IndexError(  # pragma: no cover
                f"Indexer must contain only slices or ints (received types {[type(i) for i in idx]})"
            )
        # Determine which dims should have unit size in result
        squeeze_dims = tuple(i for i in range(len(idx)) if isinstance(idx[i], int))
        # Convert all indexers to slices
        idx = tuple(slice(i, i + 1) if isinstance(i, int) else i for i in idx)

        if idx[0].start == idx[0].stop:
            return np.empty((0,) * self.ndim, dtype=self.dtype)

        # Determine start and end partitions that correspond to the
        # given variant dimension indexer
        start_partition = idx[0].start // self.partition_size
        start_partition_offset = idx[0].start % self.partition_size
        end_partition = (idx[0].stop - 1) // self.partition_size
        end_partition_offset = (idx[0].stop - 1) % self.partition_size

        # Create a list of all offsets into the underlying file at which
        # data for each variant begins
        all_vaddr = []
        with bgen_metafile(self.metafile_filepath) as mf:
            for i in range(start_partition, end_partition + 1):
                partition = mf.read_partition(i)
                start_offset = start_partition_offset if i == start_partition else 0
                end_offset = (
                    end_partition_offset + 1
                    if i == end_partition
                    else self.partition_size
                )
                vaddr = partition["vaddr"].tolist()
                all_vaddr.extend(vaddr[start_offset:end_offset])

        # Read the probabilities for each variant, apply indexer for
        # samples dimension to give probabilities for all genotypes,
        # and then apply final genotype dimension indexer
        with bgen_file(self.path) as bgen:
            res = None
            for i, vaddr in enumerate(all_vaddr):
                probs = bgen.read_genotype(vaddr)["probs"][idx[1]]
                assert len(probs.shape) == 2 and probs.shape[1] == 3
                if res is None:
                    res = np.zeros((len(all_vaddr), len(probs), 3), dtype=self.dtype)
                res[i] = probs
            res = res[..., idx[2]]
            return np.squeeze(res, axis=squeeze_dims)


def _to_dosage(probs: ArrayLike):
    """Calculate the dosage from genotype likelihoods (probabilities)"""
    assert (
        probs.shape[-1] == 3
    ), f"Expecting genotype (trailing) dimension of size 3, got array of shape {probs.shape}"
    return probs[..., 1] + 2 * probs[..., 2]


def read_bgen(
    path: PathType,
    chunks: Union[str, int, tuple] = "auto",
    lock: bool = False,
    persist: bool = True,
) -> Dataset:
    """Read BGEN dataset.

    Loads a single BGEN dataset as dask arrays within a Dataset
    from a bgen file.

    Parameters
    ----------
    path : PathType
        Path to BGEN file.
    chunks : Union[str, int, tuple], optional
        Chunk size for genotype probability data (3 dimensions),
        by default "auto".
    lock : bool, optional
        Whether or not to synchronize concurrent reads of
        file blocks, by default False. This is passed through to
        [dask.array.from_array](https://docs.dask.org/en/latest/array-api.html#dask.array.from_array).
    persist : bool, optional
        Whether or not to persist variant information in
        memory, by default True.  This is an important performance
        consideration as the metadata file for this data will
        be read multiple times when False.

    Warnings
    --------
    Only bi-allelic, diploid BGEN files are currently supported.
    """

    bgen_reader = BgenReader(path, persist)

    variant_contig, variant_contig_names = encode_array(bgen_reader.contig.compute())
    variant_contig_names = list(variant_contig_names)
    variant_contig = variant_contig.astype("int16")

    variant_position = np.array(bgen_reader.pos, dtype=int)
    variant_alleles = np.array(bgen_reader.variant_alleles, dtype=str)
    variant_id = np.array(bgen_reader.variant_id, dtype=str)

    sample_id = np.array(bgen_reader.sample_id, dtype=str)

    call_genotype_probability = da.from_array(
        bgen_reader,
        chunks=chunks,
        lock=lock,
        fancy=False,
        asarray=False,
        name=f"{bgen_reader.name}:read_bgen:{path}",
    )
    call_dosage = _to_dosage(call_genotype_probability)

    ds = create_genotype_dosage_dataset(
        variant_contig_names=variant_contig_names,
        variant_contig=variant_contig,
        variant_position=variant_position,
        variant_alleles=variant_alleles,
        sample_id=sample_id,
        call_dosage=call_dosage,
        call_genotype_probability=call_genotype_probability,
        variant_id=variant_id,
    )

    return ds


def _max_str_len(arr: ArrayLike) -> Array:
    return da.map_blocks(
        arr, lambda s: np.char.str_len(s.astype(str)), dtype=np.int8
    ).max()


def bgen_to_zarr(
    input: Union[PathType, Sequence[PathType]],
    output: PathType,
    *,
    # Using chr:pos means there is no way to get even splits since
    # positions are repeated for split alleles
    regions: Union[None, slice, Sequence[slice], Sequence[Sequence[slice]]] = None,
    chunk_length: int = 10_000,
    chunk_width: int = 1_000,
    tempdir: Optional[PathType] = None,
    compute: bool = True,
) -> None:
    # TODO
    # - Make output a store object
    # - Make sure to test fixed len str on all results
    # - Use normalize chunks in example, so it is clear to create overall array chunks
    #   https://github.com/dask/dask/blob/d96fed990279fcd272cc5ada005d07c85b104ce5/dask/array/core.py#L2357
    #   da.normalize_chunks('auto', shape=..., dtype=...) -> tuples
    # - Add mypy?

    if isinstance(input, str) or isinstance(input, Path):
        inputs: Sequence[PathType] = [input]
    else:
        inputs = input

    if regions is None:
        input_regions = [[slice(None)]] * len(inputs)
    elif isinstance(regions, slice):
        input_regions = [[regions]] * len(inputs)
    else:
        if len(regions) != len(inputs):
            raise ValueError(
                "When providing multiple input files as well as regions within them to convert, "
                "the number of regions must equal the number of input files "
                f"(received {len(inputs)} files and {len(regions)} regions)."
            )
        input_regions = regions

    assert len(inputs) == len(input_regions)

    if not tempdir:
        tempdir = Path(tempfile.mkdtemp(prefix="bgen_to_zarr_"))

    datasets = []
    parts = []
    for i, input in enumerate(inputs):
        filename = Path(input).name
        for r, region in enumerate(input_regions[i]):
            part = tempdir / filename / f"part-{r}.zarr"
            parts.append(part)
            intermediate_chunk_length = 100
            ds = read_bgen(input, chunks=(intermediate_chunk_length, -1, -1))
            ds = ds.isel(variants=region)
            encoding = {
                v: dict(chunks=dict(samples=chunk_width))
                for v in ds
                if "samples" in ds[v].dims
            }
            for v in STRING_VARS:
                ds.attrs[f"max_{v}_length"] = int(_max_str_len(da.asarray(ds[v])))
            ds = ds.to_zarr(part, mode="w", encoding=encoding, compute=False)
            datasets.append(ds)
    dask.compute(*datasets)

    datasets = [xr.open_zarr(part, overwrite_encoded_chunks=False) for part in parts]
    ds = xr.concat(datasets, dim="variants", data_vars="minimal")  # type: ignore[no-untyped-call, no-redef]
    ds = ds.chunk(dict(variants=chunk_length, samples=chunk_width))
    str_lens = {
        v: max(ds.attrs[f"max_{v}_length"] for ds in datasets) for v in STRING_VARS
    }
    for v, length in str_lens.items():
        ds[v] = ds[v].astype(f"S{length}")
        del ds.attrs[v]

    # Ensure Dask task graph is efficient, see https://github.com/dask/dask/issues/5105
    with dask.config.set({"optimization.fuse.ave-width": 50}):
        return ds.to_zarr(output, mode="w", compute=compute)
