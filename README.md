# xpublish-zarr

Zarr-compatible REST API plugin for [Xpublish](https://github.com/xpublish-community/xpublish).

`xpublish-zarr` exposes an `xarray.Dataset` over HTTP as a consolidated Zarr v3 store,
so clients can read it via `xarray.open_zarr` / `zarr.open_consolidated` over `fsspec`'s
HTTP backend.

This plugin is extracted from `xpublish` core so that the core can be used without
pulling in zarr.

> **Note on consolidated metadata.** v3 consolidated metadata is currently an
> experimental extension — not yet part of the official Zarr v3 spec, but supported by
> both zarr-python and xarray. The plugin emits it because it lets clients open the
> dataset in a single HTTP request and because dropping it would require a
> directory-listing endpoint (zarr clients need *some* way to discover the group's
> children, and HTTP doesn't have a portable listing protocol). If/when the spec lands
> a different format, we'll follow upstream.

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

ds = xr.open_zarr("http://0.0.0.0:9000/zarr/", consolidated=True, zarr_format=3)
```

### Endpoints

For a single dataset (mounted at the root):

| Path                       | Description                                              |
| -------------------------- | -------------------------------------------------------- |
| `/zarr/zarr.json`          | Root group metadata (includes consolidated metadata)     |
| `/zarr/{var}/zarr.json`    | Per-array metadata                                       |
| `/zarr/{var}/c/{i}/{j}/…`  | A variable's chunk (binary), `/`-separated chunk coords  |
| `/zarr/{var}/c`            | The single chunk of a scalar (0-d) variable              |

For a multi-dataset `Rest` app the paths are prefixed with `/datasets/{dataset_id}`.

### Caveats

- **Single group only.** DataTree / nested groups are not yet supported; this will land
  once xpublish merges its DataTree handling.
- **No sharding yet.** Adding it later is a one-line change at the `to_zarr` step and
  does not alter the HTTP surface.
- **Encoding.** Each variable's codec pipeline, chunk shape, fill value, and
  `dimension_names` come from xarray's own `to_zarr` path, so the wire format matches
  what `ds.to_zarr(..., zarr_format=3)` would produce locally. Set per-variable encoding
  (compressor, chunks, fill value, dtype) via `ds[var].encoding` before serving.

## Development

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.

```sh
uv sync
uv run pytest
```
