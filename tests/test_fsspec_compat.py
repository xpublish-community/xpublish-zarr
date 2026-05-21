import json

import pytest
from fastapi.testclient import TestClient
from zarr.buffer import default_buffer_prototype
from zarr.storage import MemoryStore

from .utils import TestStore, make_rest


async def test_get_root_zarr_json(airtemp_ds):
    client = TestClient(make_rest(airtemp_ds).app)
    store = TestStore(client)
    payload = await store.get("zarr.json", default_buffer_prototype())
    actual = json.loads(payload.to_bytes().decode())

    expected_store = MemoryStore()
    airtemp_ds.to_zarr(expected_store, zarr_format=3, consolidated=True, compute=False)
    expected_buf = await expected_store.get("zarr.json", default_buffer_prototype())
    expected = json.loads(expected_buf.to_bytes().decode())

    assert actual == expected


async def test_missing_key_raises_keyerror(airtemp_ds):
    client = TestClient(make_rest(airtemp_ds).app)
    store = TestStore(client)
    with pytest.raises(KeyError):
        _ = await store.get("notakey", default_buffer_prototype())
