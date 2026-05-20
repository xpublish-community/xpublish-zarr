"""Zarr encoding/decoding helpers used by the plugin."""

import base64
import copy
import logging
import numbers
from types import SimpleNamespace
from typing import Any, Optional, Tuple, Union, cast

import cachey
import dask.array
import numpy as np
import xarray as xr
from numcodecs.abc import Codec
from numcodecs.blosc import Blosc
from numcodecs.compat import ensure_ndarray
from xarray.backends.chunks import validate_grid_chunks_alignment
from xarray.backends.zarr import (
    DIMENSION_KEY,
    encode_zarr_attr_value,
    encode_zarr_variable,
    extract_zarr_variable_encoding,
)
from xpublish.utils.api import DATASET_ID_ATTR_KEY

DaskArrayType = (dask.array.Array,)
ZARR_FORMAT = 2
ZARR_CONSOLIDATED_FORMAT = 1
ZARR_METADATA_KEY = '.zmetadata'

# v2 store keys
array_meta_key = '.zarray'
group_meta_key = '.zgroup'
attrs_key = '.zattrs'

# Default compressor used by xarray for zarr 2 encoding
default_compressor = Blosc()

logger = logging.getLogger('xpublish_zarr')


def normalize_shape(shape: Union[int, Tuple[int, ...], None]) -> Tuple[int, ...]:
    """Convenience function to normalize the `shape` argument."""
    if shape is None:
        raise TypeError('shape is None')

    if isinstance(shape, numbers.Integral):
        shape = (int(shape),)

    shape = cast(Tuple[int, ...], shape)
    return tuple(int(s) for s in shape)


def get_zvariables(dataset: xr.Dataset, cache: cachey.Cache) -> dict:
    """Return a dictionary of zarr encoded variables, using the cache when possible."""
    cache_key = dataset.attrs.get(DATASET_ID_ATTR_KEY, '') + '/zvariables'
    zvariables = cache.get(cache_key)

    if zvariables is None:
        zvariables = create_zvariables(dataset)
        # we want to permanently cache this: set high cost value
        cache.put(cache_key, zvariables, 99999)

    return zvariables


def get_zmetadata(dataset: xr.Dataset, cache: cachey.Cache, zvariables: dict) -> dict:
    """Return a consolidated zmetadata dictionary, using the cache when possible."""
    cache_key = dataset.attrs.get(DATASET_ID_ATTR_KEY, '') + '/' + ZARR_METADATA_KEY
    zmeta = cache.get(cache_key)

    if zmeta is None:
        zmeta = create_zmetadata(dataset)
        cache.put(cache_key, zmeta, 99999)

    return zmeta


def _extract_dataset_zattrs(dataset: xr.Dataset) -> dict:
    """Create the zattrs dictionary from Dataset global attrs."""
    zattrs = {k: encode_zarr_attr_value(v) for k, v in dataset.attrs.items()}
    zattrs.pop(DATASET_ID_ATTR_KEY, None)
    return zattrs


def _extract_dataarray_zattrs(da: xr.DataArray) -> dict:
    """Extract the zattrs dictionary from a DataArray."""
    zattrs = {k: encode_zarr_attr_value(v) for k, v in da.attrs.items()}
    zattrs[DIMENSION_KEY] = list(da.dims)
    # `_FillValue` belongs in `.zarray`, not `.zattrs`
    zattrs.pop('_FillValue', None)
    return zattrs


def _extract_dataarray_coords(da: xr.DataArray, zattrs: dict) -> dict:
    """Extract non-dimension coords from a DataArray into the zattrs dict."""
    if da.coords:
        nondim_coords = set(da.coords) - set(da.dims)
        if len(nondim_coords) > 0 and da.name not in nondim_coords:
            coords = ' '.join(sorted(nondim_coords))
            zattrs['coordinates'] = encode_zarr_attr_value(coords)
    return zattrs


def _extract_fill_value(da: xr.DataArray, dtype: np.dtype) -> Any:
    """Extract the fill value from a DataArray."""
    fill_value = da.attrs.pop('_FillValue', None)
    return encode_fill_value(fill_value, dtype)


def _extract_zarray(da: xr.DataArray, encoding: dict, dtype: np.dtype) -> dict:
    """Build the zarr array metadata."""
    meta = {
        'compressor': encoding.get('compressor', da.encoding.get('compressor', default_compressor)),
        'filters': encoding.get('filters', da.encoding.get('filters', None)),
        'chunks': list(da.data.chunksize if isinstance(da.data, DaskArrayType) else da.shape),
        'dtype': dtype.str,
        'fill_value': _extract_fill_value(da, dtype),
        'order': 'C',
        'shape': list(normalize_shape(da.shape)),
        'zarr_format': ZARR_FORMAT,
        'dimension_separator': '.',
    }

    if isinstance(meta['filters'], (list, tuple)) and len(meta['filters']) == 0:
        meta['filters'] = None

    return meta


def create_zvariables(dataset: xr.Dataset) -> dict:
    """Build a dictionary of zarr-encoded variables."""
    return {key: encode_zarr_variable(da, name=key) for key, da in dataset.variables.items()}


def create_zmetadata(dataset: xr.Dataset) -> dict:
    """Build a consolidated zmetadata dictionary."""
    zmeta: dict = {
        'zarr_consolidated_format': ZARR_CONSOLIDATED_FORMAT,
        'metadata': {},
    }
    zmeta['metadata'][group_meta_key] = {'zarr_format': ZARR_FORMAT}
    zmeta['metadata'][attrs_key] = _extract_dataset_zattrs(dataset)

    for key, dvar in dataset.variables.items():
        da = dataset[key]

        # If the variable is a dask_array with non-uniform chunks, rechunk it.
        if isinstance(dvar.data, DaskArrayType) and any(
            (len(set(chunks[:-1])) > 1 or chunks[0] < chunks[-1]) for chunks in dvar.data.chunks
        ):
            da.variable.data = da.variable.data.rechunk(dvar.data.chunksize)
            dvar = da.variable

        encoded_da = encode_zarr_variable(dvar, name=key)
        if 'chunks' in encoded_da.encoding:
            validate_grid_chunks_alignment(
                nd_v_chunks=dvar.chunks,
                enc_chunks=encoded_da.encoding['chunks'],
                region=tuple(SimpleNamespace(start=None, stop=None) for _ in da.shape),
                allow_partial_chunks=False,
                name=key,
                backend_shape=encoded_da.shape,
            )
        encoding = extract_zarr_variable_encoding(dvar, zarr_format=ZARR_FORMAT)
        zattrs = _extract_dataarray_zattrs(encoded_da)
        zattrs = _extract_dataarray_coords(da, zattrs)
        zmeta['metadata'][f'{key}/{attrs_key}'] = zattrs
        zmeta['metadata'][f'{key}/{array_meta_key}'] = _extract_zarray(
            encoded_da,
            encoding,
            encoded_da.dtype,
        )

    return zmeta


def jsonify_zmetadata(dataset: xr.Dataset, zmetadata: dict) -> dict:
    """Convert the zmetadata dictionary to a json-compatible dictionary."""
    zjson = copy.deepcopy(zmetadata)

    for key in list(dataset.variables):
        zarray = zjson['metadata'][f'{key}/{array_meta_key}']

        compressor = zarray['compressor']
        if compressor is not None:
            zarray['compressor'] = compressor.get_config()

        filters = zarray['filters']
        if filters is not None:
            zarray['filters'] = [f.get_config() for f in filters]

    return zjson


def encode_chunk(
    chunk: np.typing.ArrayLike,
    filters: Optional[list[Codec]] = None,
    compressor: Optional[Codec] = None,
) -> np.typing.ArrayLike:
    """Encode a chunk by applying filters and the compressor."""
    if filters:
        for f in filters:
            chunk = f.encode(chunk)

    if ensure_ndarray(chunk).dtype == object:
        raise RuntimeError('cannot write object array without object codec')

    if compressor:
        return compressor.encode(chunk)
    return chunk


def get_data_chunk(
    da: xr.DataArray,
    chunk_id: str,
    out_shape: tuple,
) -> np.typing.ArrayLike:
    """Get one chunk of data from this DataArray (da).

    If this is an incomplete edge chunk, pad the returned array to match out_shape.
    """
    ikeys = tuple(map(int, chunk_id.split('.')))
    if isinstance(da, DaskArrayType):
        chunk_data = da.blocks[ikeys]
    else:
        if da.ndim > 0 and ikeys != ((0,) * da.ndim):
            raise ValueError(
                f'Invalid chunk_id for numpy array: {chunk_id}. Should have been: {(0,) * da.ndim}',
            )
        chunk_data = np.asarray(da)

    logger.debug('checking chunk output size, %s == %s', chunk_data.shape, out_shape)

    if isinstance(chunk_data, DaskArrayType):
        chunk_data = chunk_data.compute()

    # zarr expects full edge chunks; contents out of bounds for the array are undefined
    if chunk_data.shape != tuple(out_shape):
        new_chunk = np.empty_like(chunk_data, shape=out_shape)
        write_slice = tuple(slice(0, s) for s in chunk_data.shape)
        new_chunk[write_slice] = chunk_data
        return new_chunk
    return chunk_data


def encode_fill_value(v: Any, dtype: np.dtype, object_codec: Any = None) -> Any:
    """Encode a fill value for a zarr array."""
    if v is None:
        return None
    if dtype.kind in 'SV':
        # bytes-type fill value
        return base64.standard_b64encode(cast(bytes, v)).decode('ascii')
    if isinstance(v, np.datetime64):
        return np.datetime_as_string(v)
    if isinstance(v, numbers.Integral):
        return int(v)
    if isinstance(v, numbers.Real):
        float_fv = float(v)
        if np.isnan(float_fv):
            return 'NaN'
        if np.isinf(float_fv):
            return 'Infinity' if float_fv > 0 else '-Infinity'
        return float_fv
    if isinstance(v, numbers.Complex):
        return [encode_fill_value(v.real, dtype), encode_fill_value(v.imag, dtype)]
    return v
