"""Test utilities for the xpublish-zarr plugin."""

from functools import reduce
from operator import mul
from typing import AsyncIterator, Iterable

import numpy as np
import pandas as pd
import xarray as xr
from starlette.testclient import TestClient
from xpublish import SingleDatasetRest
from zarr.abc.store import ByteRequest, Store
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.core.common import BytesLike

from xpublish_zarr import ZarrPlugin

rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(123456789)))


def make_rest(dataset: xr.Dataset) -> SingleDatasetRest:
    """Build a SingleDatasetRest wired up to our local ZarrPlugin only."""
    return SingleDatasetRest(dataset, plugins={'zarr': ZarrPlugin()})


class TestStore(Store):
    """A simple zarr Store backed by a Starlette TestClient."""

    __test__ = False  # Tell pytest not to collect this as a test class.

    _client: TestClient

    def __init__(self, client: TestClient):
        self._client = client
        self._is_open = True
        self._read_only = True

    def __eq__(self, value: object) -> bool:
        """Equality comparison."""
        return isinstance(value, type(self)) and self._client == value._client

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        """Retrieve the value associated with a given key."""
        zarr_key = f'/zarr/{key}'
        response = self._client.get(zarr_key)
        if response.status_code != 200:
            raise KeyError(f'{zarr_key} not found. status_code = {response.status_code}')
        return prototype.buffer.from_bytes(response.content)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        """Retrieve possibly partial values from given key_ranges."""
        return NotImplemented

    async def exists(self, key: str) -> bool:
        """Check if a key exists in the store."""
        return NotImplemented

    @property
    def supports_writes(self) -> bool:
        """Does the store support writes?"""
        return False

    async def set(self, key: str, value: Buffer) -> None:
        """Store a (key, value) pair."""
        return NotImplemented

    @property
    def supports_deletes(self) -> bool:
        """Does the store support deletes?"""
        return False

    async def delete(self, key: str) -> None:
        """Remove a key from the store."""
        return NotImplemented

    @property
    def supports_partial_writes(self) -> bool:
        """Does the store support partial writes?"""
        return False

    async def set_partial_values(
        self, key_start_values: Iterable[tuple[str, int, BytesLike]],
    ) -> None:
        """Store values at a given key, starting at byte range_start."""
        return NotImplemented

    def supports_listing(self) -> bool:
        """Does the store support listing?"""
        return True

    async def list(self) -> AsyncIterator[str]:
        """Retrieve all keys in the store."""
        return NotImplemented

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        """Retrieve all keys in the store that begin with a given prefix."""
        return NotImplemented

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        """Retrieve all keys and prefixes with a given prefix not containing further '/'."""
        return NotImplemented


def create_dataset(
    start='2018-01',
    end='2020-12',
    freq='MS',
    calendar='standard',
    units='days since 1980-01-01',
    use_cftime=True,
    decode_times=True,
    nlats=1,
    nlons=1,
    var_const=None,
    use_xy_dim=False,
):
    """Build a synthetic xarray Dataset for tests."""
    if use_cftime:
        end = xr.coding.cftime_offsets.to_cftime_datetime(end, calendar=calendar)
        dates = xr.date_range(
            start=start, end=end, freq=freq, calendar=calendar, use_cftime=use_cftime,
        )
    else:
        dates = pd.date_range(start=pd.to_datetime(start), end=pd.to_datetime(end), freq=freq)

    decoded_time_bounds = np.vstack((dates[:-1], dates[1:])).T

    encoded_time_bounds = xr.coding.times.encode_cf_datetime(
        decoded_time_bounds, units=units, calendar=calendar,
    )[0]

    encoded_times = xr.DataArray(
        encoded_time_bounds.mean(axis=1),
        dims=('time'),
        name='time',
        attrs={'units': units, 'calendar': calendar},
    )

    decoded_times = xr.DataArray(
        xr.coding.times.decode_cf_datetime(
            encoded_times, units=units, calendar=calendar, use_cftime=use_cftime,
        ),
        dims=['time'],
    )
    decoded_time_bounds = xr.DataArray(
        decoded_time_bounds,
        name='time_bounds',
        dims=('time', 'd2'),
        coords={'time': decoded_times},
    )

    if decode_times:
        times = decoded_times
        time_bounds = decoded_time_bounds
    else:
        times = encoded_times
        time_bounds = xr.DataArray(
            encoded_time_bounds,
            name='time_bounds',
            dims=('time', 'd2'),
            coords={'time': times},
        )

    lats = np.linspace(start=-90, stop=90, num=nlats, dtype='float32')
    lons = np.linspace(start=-180, stop=180, num=nlons, dtype='float32')

    if use_xy_dim:
        lats, lons = np.meshgrid(lats, lons)
        shape = (times.size, *lats.shape)
    else:
        shape = (times.size, lats.size, lons.size)

    num = reduce(mul, shape)

    if var_const is None:
        annual_cycle = np.sin(2 * np.pi * (decoded_times.dt.dayofyear.values / 365.25 - 0.28))
        base = 10 + 15 * annual_cycle.reshape(-1, 1)
        tmin_values = base + 3 * np.random.randn(annual_cycle.size, nlats * nlons)
        tmax_values = base + 10 + 3 * np.random.randn(annual_cycle.size, nlats * nlons)
        tmin_values = tmin_values.reshape(shape)
        tmax_values = tmax_values.reshape(shape)
    elif var_const:
        tmin_values = np.ones(shape=shape)
        tmax_values = np.ones(shape=shape) + 2
    else:
        tmin_values = np.arange(1, num + 1).reshape(shape)
        tmax_values = np.arange(1, num + 1).reshape(shape)

    ds = xr.Dataset(
        {
            'tmin': xr.DataArray(
                tmin_values.astype('float32'),
                dims=('time', 'lat', 'lon') if not use_xy_dim else ('time', 'y', 'x'),
                name='tmin',
            ),
            'tmax': xr.DataArray(
                tmax_values.astype('float32'),
                dims=('time', 'lat', 'lon') if not use_xy_dim else ('time', 'y', 'x'),
                name='tmax',
            ),
            'time_bounds': time_bounds,
        },
        coords={
            'time': times,
            'lat': ('lat' if not use_xy_dim else ['y', 'x'], lats),
            'lon': ('lon' if not use_xy_dim else ['y', 'x'], lons),
        },
    )

    ds.tmin.encoding['_FillValue'] = np.float32(-9999999)
    ds.tmax.encoding['_FillValue'] = np.float32(-9999999)
    ds.time.attrs['bounds'] = 'time_bounds'
    ds.time.encoding['units'] = units
    ds.time.encoding['calendar'] = calendar

    return ds
