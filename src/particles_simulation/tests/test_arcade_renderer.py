import importlib

import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("arcade") is None, reason="arcade not installed"
)
def test_arcade_module_imports():
    """Sanity test: module imports and exposes `ArcadeRenderer` and `run_arcade_demo`."""
    mod = importlib.import_module("particles_simulation.arcade_renderer")
    assert hasattr(mod, "ArcadeRenderer")
    assert hasattr(mod, "run_arcade_demo")
