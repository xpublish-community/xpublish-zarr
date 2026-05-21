"""Zarr v3 store helpers used by the plugin."""

import asyncio
import logging
import warnings

import cachey
import numpy as np
import xarray as xr
import zarr.api.asynchronous
from xarray.backends.zarr import encode_zarr_variable
from xpublish.utils.api import DATASET_ID_ATTR_KEY
from zarr.buffer import default_buffer_prototype
from zarr.storage import MemoryStore

logger = logging.getLogger("xpublish_zarr")

ROOT_METADATA_KEY = "zarr.json"
ARRAY_METADATA_SUFFIX = "/zarr.json"


def _build_store(dataset: xr.Dataset) -> MemoryStore:
    """Write a v3 metadata-only zarr hierarchy for the dataset.

    Dask-backed data variables are deferred (encoded on demand per chunk
    request). Numpy-backed variables (including coords) are materialized.
    """
    store = MemoryStore()
    with warnings.catch_warnings():
        # xarray emits a UserWarning that v3 consolidated metadata isn't in
        # the official spec yet. We've made the deliberate choice to use it
        # for fast metadata loads over HTTP, so silence here.
        warnings.filterwarnings(
            "ignore",
            message="Consolidated metadata is currently not part",
        )
        dataset.to_zarr(store, zarr_format=3, compute=False, consolidated=True)
    return store


def get_store(dataset: xr.Dataset, cache: cachey.Cache) -> MemoryStore:
    """Return the cached v3 MemoryStore for this dataset, building it if needed."""
    cache_key = dataset.attrs.get(DATASET_ID_ATTR_KEY, "") + "/zarr-v3-store"
    store = cache.get(cache_key)
    if store is None:
        store = _build_store(dataset)
        cache.put(cache_key, store, 99999)
    return store


def get_bytes(store: MemoryStore, key: str) -> bytes | None:
    """Read raw bytes for a key from the store."""
    buf = store.get_sync(key, prototype=default_buffer_prototype())
    return buf.to_bytes() if buf is not None else None


def encode_chunk(
    dataset: xr.Dataset,
    metadata_store: MemoryStore,
    var: str,
    chunk_coords: tuple[int, ...],
) -> bytes:
    """Encode one chunk of `var` through zarr's codec pipeline.

    Runs the encode under a private asyncio loop to avoid re-entering
    zarr's global sync event loop when this code is reached from inside
    an outer `xr.open_zarr` call (e.g. in tests under TestClient).
    """
    if var not in dataset.variables:
        raise KeyError(var)

    # Re-encode the variable so its values match the on-disk dtype expected
    # by the zarr array (e.g. cftime datetimes → int64 with units/calendar).
    encoded = encode_zarr_variable(dataset.variables[var], name=var)
    return asyncio.run(_encode_chunk_async(encoded, metadata_store, var, chunk_coords))


async def _encode_chunk_async(
    var_data: xr.Variable,
    metadata_store: MemoryStore,
    var: str,
    chunk_coords: tuple[int, ...],
) -> bytes:
    array_meta_key = f"{var}{ARRAY_METADATA_SUFFIX}"
    scratch = MemoryStore()
    meta_buf = await metadata_store.get(array_meta_key, prototype=default_buffer_prototype())
    if meta_buf is None:
        raise KeyError(array_meta_key)
    await scratch.set(array_meta_key, meta_buf)

    aarr = await zarr.api.asynchronous.open_array(store=scratch, path=var, mode="r+")
    chunk_shape = aarr.chunks
    shape = aarr.shape

    if len(chunk_coords) != len(chunk_shape):
        raise IndexError(
            f"chunk {chunk_coords} has wrong rank for {var} (expected {len(chunk_shape)} indices)",
        )

    slices = tuple(
        slice(c * cs, min((c + 1) * cs, s)) for c, cs, s in zip(chunk_coords, chunk_shape, shape)
    )
    if chunk_shape and any(sl.start >= sl.stop for sl in slices):
        raise IndexError(f"chunk {chunk_coords} out of range for {var} shape={shape}")

    if chunk_shape:
        chunk_data = np.asarray(
            var_data.isel({d: sl for d, sl in zip(var_data.dims, slices)}).values
        )
        await aarr.setitem(slices, chunk_data)
    else:
        await aarr.setitem(..., np.asarray(var_data.values))

    chunk_key = chunk_storage_key(var, chunk_coords)
    buf = await scratch.get(chunk_key, prototype=default_buffer_prototype())
    if buf is None:
        raise RuntimeError(f"encoded chunk missing from scratch store: {chunk_key}")
    return buf.to_bytes()


def chunk_storage_key(var: str, chunk_coords: tuple[int, ...]) -> str:
    """Build the store key for a chunk, matching v3 default chunk_key_encoding."""
    if not chunk_coords:
        return f"{var}/c"
    return f"{var}/c/" + "/".join(str(c) for c in chunk_coords)
