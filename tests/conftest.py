"""Pytest fixtures and configuration for HealICON tests."""
import healpy  # noqa: F401 - initialise astropy logger before coverage warning redirection
import pytest
from dask.callbacks import Callback


@pytest.fixture(autouse=True)
def _clean_dask_callbacks():
    """Ensure no stale Dask callbacks leak between tests."""
    yield
    Callback.active.clear()
