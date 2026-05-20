# xpublish-zarr

Zarr-compatible REST API plugin for [Xpublish](https://github.com/xpublish-community/xpublish).

`xpublish-zarr` exposes an `xarray.Dataset` over HTTP as a consolidated Zarr v2 store,
so clients can read it via `xarray.open_zarr` / `zarr.open_consolidated` over `fsspec`'s
HTTP backend.

This plugin is extracted from `xpublish` core so that the core can be used without
pulling in zarr.

## Install

```sh
uv add xpublish-zarr
# or
pip install xpublish-zarr
```

The plugin is auto-registered with xpublish via the `xpublish.plugin` entry point.

## Usage

```python
import xarray as xr

ds = xr.tutorial.open_dataset("air_temperature")
ds.rest.serve(host="0.0.0.0", port=9000)
```

```python
import xarray as xr

ds = xr.open_zarr("http://0.0.0.0:9000/zarr/", consolidated=True, zarr_format=2)
```

### Endpoints

For a single dataset (mounted at the root):

| Path             | Description                              |
| ---------------- | ---------------------------------------- |
| `/zarr/.zmetadata` | Consolidated zarr metadata             |
| `/zarr/.zgroup`    | Zarr group metadata                    |
| `/zarr/.zattrs`    | Dataset attributes                     |
| `/zarr/{var}/{chunk}` | A variable's chunk (binary)         |

For a multi-dataset `Rest` app the paths are prefixed with `/datasets/{dataset_id}`.

## Development

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.

```sh
uv sync
uv run pytest
```
