"""BGEN reader implementation (using bgen_reader)"""
import tempfile
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import xarray as xr
import zarr
from bgen_reader._bgen_file import bgen_file
from bgen_reader._bgen_metafile import bgen_metafile
from bgen_reader._metafile import create_metafile
from bgen_reader._reader import infer_metafile_filepath
from bgen_reader._samples import generate_samples, read_samples_file
from dask import Array
from xarray import Dataset
from zarr import ZarrStore

from sgkit import create_genotype_dosage_dataset
from sgkit.typing import ArrayLike
from sgkit.utils import encode_array

PathType = Union[str, Path]

STRING_VARS = ["variant_id", "variant_allele", "sample_id"]


def _to_dict(df: dd.DataFrame, dtype: Any = None) -> Dict[str, da.Array]:
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

    def __init__(
        self, path: PathType, persist: bool = True, dtype: Any = np.float32
    ) -> None:
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

            def split_alleles(
                alleles: np.ndarray, block_info: Any = None
            ) -> np.ndarray:
                if block_info is None or len(block_info) == 0:
                    return alleles

                def split(allele_row: np.ndarray) -> np.ndarray:
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

    def __getitem__(self, idx: Any) -> np.ndarray:
        if not isinstance(idx, tuple):
            raise IndexError(f"Indexer must be tuple (received {type(idx)})")
        if len(idx) != self.ndim:
            raise IndexError(
                f"Indexer must have {self.ndim} items (received {len(idx)} slices)"
            )
        if not all(isinstance(i, slice) or isinstance(i, int) for i in idx):
            raise IndexError(
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
            res = res[..., idx[2]]  # type: ignore[index]
            return np.squeeze(res, axis=squeeze_dims)


def _to_dosage(probs: ArrayLike) -> ArrayLike:
    """Calculate the dosage from genotype likelihoods (probabilities)"""
    assert (
        probs.shape[-1] == 3
    ), f"Expecting genotype (trailing) dimension of size 3, got array of shape {probs.shape}"
    return probs[..., 1] + 2 * probs[..., 2]


def read_bgen(
    path: PathType,
    chunks: Union[str, int, Tuple[int, ...]] = "auto",
    lock: bool = False,
    persist: bool = True,
    dtype: Any = "float32",
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
    dtype : Any
        Genotype probability array data type, by default float32.

    Warnings
    --------
    Only bi-allelic, diploid BGEN files are currently supported.
    """

    bgen_reader = BgenReader(path, persist, dtype=dtype)

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

    ds: Dataset = create_genotype_dosage_dataset(
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


def _bgen_to_zarr(
    input: Path,
    output: Path,
    region: slice,
    chunk_length: int,
    chunk_width: int,
    target_chunk_width: int,
    compressor: Any,
    read_fn: Callable[[Path, Tuple[int, int, int]], Dataset],
):
    ds = read_fn(path=input, chunks=(chunk_length, chunk_width, -1))

    # Apply slice to region
    ds = ds.isel(variants=region)

    # Remove dosage/gp mask as they are unnecessary and should be redefined
    # based on encoded probabilities later (w/ reduced precision)
    ds = ds.drop_vars(
        ["call_dosage", "call_dosage_mask", "call_genotype_probability_mask"],
        errors="ignore",
    )

    # Slice off homozygous ref GP and redefine mask
    gp = ds["call_genotype_probability"][..., 1:]
    gp_mask = np.isnan(gp).any(dim="genotypes")
    ds = ds.drop_vars(["call_genotype_probability"])
    ds = ds.assign(
        call_genotype_probability=gp, call_genotype_probability_mask=gp_mask,
    )

    # Set compressor, chunking and floating point encoding
    def var_encoding(ds, v):
        e = {"compressor": compressor}
        if "samples" in ds[v].dims:
            e["chunks"] = dict(samples=target_chunk_width)
        if v == "call_genotype_probability":
            e.update(
                {
                    "dtype": "uint8",
                    "add_offset": -1.0 / 254.0,
                    "scale_factor": 1.0 / 254.0,
                    "_FillValue": 0,
                }
            )
        return e

    encoding = {v: var_encoding(ds, v) for v in ds}
    ds.to_zarr(output, mode="w", encoding=encoding, compute=True)


def bgen_to_zarrs(
    input: Union[PathType, Sequence[PathType]],
    output: PathType,
    *,
    regions: Union[None, slice, Sequence[slice], Sequence[Sequence[slice]]] = None,
    chunk_length: int = 100,
    chunk_width: int = -1,
    target_chunk_width: int = 10_000,
    compressor: Any = zarr.Blosc(cname="zstd", clevel=7, shuffle=2),
    read_fn: Callable[[Path, Tuple[int, int, int]], Dataset] = read_bgen,
) -> Sequence[Path]:
    output = Path(output)

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

    parts = []
    for i, input in enumerate(inputs):
        filename = Path(input).name
        for r, region in enumerate(input_regions[i]):
            part = output / filename / f"part-{r}.zarr"
            parts.append(part)
            _bgen_to_zarr(
                input=input,
                output=part,
                region=region,
                chunk_length=chunk_length,
                chunk_width=chunk_width,
                target_chunk_width=target_chunk_width,
                compressor=compressor,
                read_fn=read_fn,
            )
    return parts


def zarrs_to_dataset(
    paths: Sequence[Path],
    chunk_length: int = 10_000,
    chunk_width: int = 10_000,
    string_vars: Sequence[Hashable] = STRING_VARS,
    mask_and_scale: bool = True,
) -> Dataset:
    datasets = [xr.open_zarr(path, mask_and_scale=mask_and_scale) for path in paths]
    ds = xr.concat(datasets, dim="variants", data_vars="minimal")  # type: ignore[no-untyped-call, no-redef]
    ds = ds.chunk(dict(variants=chunk_length, samples=chunk_width))
    for v in string_vars:
        length = int(_max_str_len(ds[v]))
        ds[v] = ds[v].astype(f"S{length}")
    return ds


def bgen_to_zarr(
    input: Union[PathType, Sequence[PathType]],
    store: Union[PathType, MutableMapping],
    *,
    regions: Union[None, slice, Sequence[slice], Sequence[Sequence[slice]]] = None,
    chunk_length: int = 10_000,
    chunk_width: int = 1_000,
    temp_chunk_length: int = 100,
    temp_chunk_width: int = -1,
    temp_dir: Optional[PathType] = None,
    read_fn: Callable[[PathType, Tuple[int, int, int]], Dataset] = read_bgen,
) -> ZarrStore:
    """Convert BGEN to Zarr.

    This operation works in two phases:

    1. Load raw bgen for very small variant chunks and very large sample
        chunks (all of them by default), and write out as temporary zarr
        datasets containing the desired sample chunking.
    2. Rechunk the temporary result with the desired variant chunking.

    The inputs and temporary results must be local, but the final output
    store can be local or remote.

    Parameters
    ----------
    input : Union[PathType, Sequence[PathType]]
        [description]
    store : Union[MutableMapping, str, pathlib.Path]
        [description]
    chunk_length : int, optional
        [description], by default 10_000
    chunk_width : int, optional
        [description], by default 1_000
    temp_chunk_length : int, optional
        [description], by default 100
    temp_chunk_width : int, optional
        [description], by default -1
    temp_dir : Optional[PathType], optional
        [description], by default None
    compute : bool, optional
        [description], by default True

    Returns
    -------
    ZarrStore
        [description]

    Raises
    ------
    ValueError
        [description]
    """
    if not temp_dir:
        temp_dir = Path(tempfile.mkdtemp(prefix="bgen_to_zarr_"))

    paths = bgen_to_zarrs(
        input,
        temp_dir,
        regions=regions,
        chunk_length=temp_chunk_length,
        chunk_width=temp_chunk_width,
        read_fn=read_fn,
    )

    ds = zarrs_to_dataset(
        paths, chunk_length=chunk_length, chunk_width=chunk_width, mask_and_scale=False
    )

    # Ensure Dask task graph is efficient since there are so many small chunks
    # in the temporary results, see https://github.com/dask/dask/issues/5105
    with dask.config.set({"optimization.fuse.ave-width": 50}):
        return ds.to_zarr(store, mode="w", compute=True)
