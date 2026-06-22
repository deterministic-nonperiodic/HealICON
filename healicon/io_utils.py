import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import dask
import xarray as xr
from dask.base import is_dask_collection

try:
    # pyrefly: ignore [missing-import]
    import zarr as _zarr
    _ZARR_AVAILABLE = True
except ImportError:
    _ZARR_AVAILABLE = False

logger = logging.getLogger(__name__)

_global_attrs = {
    'source': 'git@github.com:deterministic-nonperiodic/HealICON.git',
    'institution': 'Leibniz Institute for Atmospheric Physics (IAP)',
    'history': datetime.today().strftime('Created on %c'),
    'references': '',
    'Conventions': 'CF-1.6'
}


def _resolve_store_and_path(path: Path | str, store_type: Optional[str] = None) -> Tuple[Path, str]:
    if isinstance(path, str):
        path = Path(path)

    # 1. Determine Target Store
    store = (store_type or "").lower()

    # Infer store from extension if store_type is empty/none
    suffix = path.suffix.lower()

    if store not in {"netcdf", "zarr"}:
        if suffix in {".nc", ".nc4", ".cdf"}:
            store = "netcdf"
        elif suffix == ".zarr":
            store = "zarr"
        else:
            # Default preference if nothing is recognized (NetCDF is default for HealICON)
            store = "netcdf"
            if suffix != "":
                logger.info(f"Unrecognized output extension '{suffix}' -- defaulting to NetCDF.")

    # 2. Ensure Extension Matches Resolved Store
    if store == "zarr":
        if suffix != ".zarr":
            path = path.with_suffix(".zarr")
            logger.info(f"[I/O] Corrected output path to {path} (Zarr).")
    elif store == "netcdf":
        if suffix not in {".nc", ".nc4", ".cdf"}:
            path = path.with_suffix(".nc")
            logger.info(f"[I/O] Corrected output path to {path} (NetCDF).")

    return path, store


def write_dataset(
    ds: xr.Dataset,
    ofile: Optional[str | Path] = None,
    overwrite: bool = True,
    store_type: Optional[str] = None,
    client=None,
    engine: str = "netcdf4",
    cfg=None
) -> None:
    """
    Write a Dask-backed xarray.Dataset to disk efficiently.

    For NetCDF output, writes directly with ``to_netcdf``, respecting any
    active Dask scheduler context set by the caller.  This is the most
    efficient path for HealICON's ``map_blocks``-based outputs, where
    computation rather than I/O is the bottleneck.

    For Zarr output (when ``zarr`` is installed and the path ends in
    ``.zarr``), writes in parallel using Dask.  If a distributed ``client``
    is provided (or can be auto-detected), the graph is submitted to the
    cluster; otherwise the local ``threads`` scheduler is used.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to write. May be Dask-backed or already in memory.
    ofile : str or Path, optional
        Destination file path. Must be provided when ``cfg`` is not.
    overwrite : bool, default True
        Whether to overwrite existing files or stores.
    store_type : str, optional
        ``'zarr'`` or ``'netcdf'``. Inferred from the file extension when
        not provided.
    client : dask.distributed.Client, optional
        Dask distributed client. Auto-detected if not provided.
    engine : str, default ``'netcdf4'``
        NetCDF engine passed to ``xarray.Dataset.to_netcdf``.
    cfg : Namespace, optional
        Legacy ``sbudget``-style config object. Reads ``cfg.output.path``,
        ``cfg.output.store``, ``cfg.output.overwrite``, and
        ``cfg.input.engine`` when provided.
    """
    # --- Resolve configuration ---
    if cfg is not None:
        output_path = Path(getattr(cfg.output, "path", "output.nc"))
        store_type = getattr(cfg.output, "store", None)
        overwrite = getattr(cfg.output, "overwrite", True)
        engine = getattr(cfg.input, "engine", "netcdf4")
    else:
        if ofile is None:
            raise ValueError("Either 'ofile' or 'cfg' must be provided to write_dataset.")
        output_path = Path(ofile)

    output_path, store_type = _resolve_store_and_path(output_path, store_type)

    # --- Overwrite safety ---
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} exists; set overwrite=True to replace")
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    ds.attrs.update(_global_attrs)

    # ── NetCDF path: write directly, honour the caller's scheduler context ──
    if store_type == "netcdf":
        logger.info(f"[I/O] Writing to {output_path} ...")
        ds.to_netcdf(output_path, engine=engine)
        logger.info(f"[I/O] Done → '{output_path}'")
        return

    # ── Zarr path: parallel write using Dask ────────────────────────────────
    if not _ZARR_AVAILABLE:
        raise ImportError(
            "zarr is required to write Zarr stores. "
            "Install it with: conda install zarr  or  pip install zarr"
        )

    logger.info(f"[I/O] Writing Zarr store to {output_path} ...")

    delayed_write_op = ds.to_zarr(
        output_path,
        mode="w",
        compute=False,
        zarr_format=3,
        consolidated=False,
    )

    if not is_dask_collection(delayed_write_op):
        delayed_write_op.compute()
        logger.info(f"[I/O] Done → '{output_path}' (Zarr)")
        return

    # Auto-detect distributed client
    if client is None:
        try:
            import dask.distributed
            client = dask.distributed.client.default_client()
        except (ValueError, ImportError):
            client = None

    if client:
        logger.info("[I/O] Executing Zarr write on distributed client ...")
        from dask.distributed import progress
        future = client.compute(delayed_write_op)
        progress(future, notebook=False)
        try:
            future.result()
        except Exception as e:
            logger.error("[I/O] FATAL: Zarr write failed on the cluster.")
            if output_path.exists():
                shutil.rmtree(output_path)
            raise
    else:
        active_scheduler = dask.config.get('scheduler', None)
        if active_scheduler is None:
            logger.info("[I/O] Executing Zarr write using threads scheduler ...")
            ctx = dask.config.set(scheduler='threads')
        else:
            import contextlib
            logger.info(f"[I/O] Executing Zarr write using scheduler: {active_scheduler} ...")
            ctx = contextlib.nullcontext()

        try:
            from dask.diagnostics import ProgressBar
            pb_ctx = ProgressBar()
        except ImportError:
            import contextlib
            pb_ctx = contextlib.nullcontext()

        with ctx, pb_ctx:
            dask.compute(delayed_write_op)

    logger.info(f"[I/O] Done → '{output_path}' (Zarr)")
