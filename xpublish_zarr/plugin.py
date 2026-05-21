"""Zarr v3 REST API plugin for Xpublish."""

import logging
from typing import Sequence

import cachey
import xarray as xr
from fastapi import APIRouter, Depends, HTTPException, Path
from starlette.responses import Response
from xpublish.plugins import Dependencies, Plugin, hookimpl
from xpublish.utils.api import DATASET_ID_ATTR_KEY
from xpublish.utils.cache import CostTimer

from xpublish_zarr.utils import (
    ARRAY_METADATA_SUFFIX,
    ROOT_METADATA_KEY,
    chunk_storage_key,
    encode_chunk,
    get_bytes,
    get_store,
)

logger = logging.getLogger("xpublish_zarr")


class ZarrPlugin(Plugin):
    """Adds Zarr v3-compatible accessing endpoints for datasets."""

    name: str = "zarr"

    dataset_router_prefix: str = "/zarr"
    dataset_router_tags: Sequence[str] = ["zarr"]

    @hookimpl
    def dataset_router(self, deps: Dependencies) -> APIRouter:
        """Build the dataset-scoped router exposing zarr v3 endpoints."""
        router = APIRouter(
            prefix=self.dataset_router_prefix,
            tags=list(self.dataset_router_tags),
        )

        @router.get("/" + ROOT_METADATA_KEY)
        def get_root_metadata(
            dataset: xr.Dataset = Depends(deps.dataset),
            cache: cachey.Cache = Depends(deps.cache),
        ):
            """Root group zarr.json (includes consolidated metadata)."""
            store = get_store(dataset, cache)
            payload = get_bytes(store, ROOT_METADATA_KEY)
            if payload is None:
                raise HTTPException(status_code=404, detail=ROOT_METADATA_KEY)
            return Response(payload, media_type="application/json")

        @router.get("/{var}" + ARRAY_METADATA_SUFFIX)
        def get_array_metadata(
            var: str = Path(description="Variable in dataset"),
            dataset: xr.Dataset = Depends(deps.dataset),
            cache: cachey.Cache = Depends(deps.cache),
        ):
            """Per-array zarr.json."""
            store = get_store(dataset, cache)
            payload = get_bytes(store, f"{var}{ARRAY_METADATA_SUFFIX}")
            if payload is None:
                raise HTTPException(status_code=404, detail=f"{var}{ARRAY_METADATA_SUFFIX}")
            return Response(payload, media_type="application/json")

        @router.get("/{var}/c/{chunk:path}")
        def get_chunk(
            var: str = Path(description="Variable in dataset"),
            chunk: str = Path(description="Chunk coordinates separated by '/'"),
            dataset: xr.Dataset = Depends(deps.dataset),
            cache: cachey.Cache = Depends(deps.cache),
        ):
            """Get an encoded zarr v3 chunk."""
            store = get_store(dataset, cache)
            chunk_coords = _parse_chunk_path(chunk)
            store_key = chunk_storage_key(var, chunk_coords)

            cache_key = dataset.attrs.get(DATASET_ID_ATTR_KEY, "") + "/v3-chunk/" + store_key
            response = cache.get(cache_key)
            if response is not None:
                return response

            with CostTimer() as ct:
                # Coords and non-dask data variables are pre-materialized in
                # the store by to_zarr(compute=False); serve those directly.
                payload = get_bytes(store, store_key)
                if payload is None:
                    if var not in dataset.variables:
                        raise HTTPException(status_code=404, detail=var)
                    try:
                        payload = encode_chunk(dataset, store, var, chunk_coords)
                    except IndexError as e:
                        raise HTTPException(status_code=404, detail=str(e)) from e
                response = Response(payload, media_type="application/octet-stream")

            cache.put(cache_key, response, ct.time, len(payload))
            return response

        @router.get("/{var}/c")
        def get_scalar_chunk(
            var: str = Path(description="Scalar variable in dataset"),
            dataset: xr.Dataset = Depends(deps.dataset),
            cache: cachey.Cache = Depends(deps.cache),
        ):
            """Get the single chunk of a scalar (0-d) variable."""
            store = get_store(dataset, cache)
            store_key = chunk_storage_key(var, ())
            cache_key = dataset.attrs.get(DATASET_ID_ATTR_KEY, "") + "/v3-chunk/" + store_key
            response = cache.get(cache_key)
            if response is not None:
                return response

            with CostTimer() as ct:
                payload = get_bytes(store, store_key)
                if payload is None:
                    if var not in dataset.variables:
                        raise HTTPException(status_code=404, detail=var)
                    payload = encode_chunk(dataset, store, var, ())
                response = Response(payload, media_type="application/octet-stream")

            cache.put(cache_key, response, ct.time, len(payload))
            return response

        return router


def _parse_chunk_path(chunk: str) -> tuple[int, ...]:
    """Parse the trailing chunk-coordinate path into integer coords."""
    if not chunk:
        return ()
    try:
        return tuple(int(part) for part in chunk.split("/"))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"invalid chunk path: {chunk}") from e
