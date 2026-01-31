"""Tests for the command-line interface."""

import pytest
from particles_simulation.cli import main


def test_cli_help():
    """Test that --help shows help message."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


def test_cli_default_run_arcade():
    """Test that default behavior runs arcade demo (dry-run)."""
    result = main(["--dry-run"])
    assert result == 0


def test_cli_no_arcade_demo():
    """Test --no-arcade-demo flag skips arcade."""
    result = main(["--no-arcade-demo"])
    assert result == 0


def test_cli_custom_n_particles():
    """Test custom number of particles argument."""
    result = main(["--dry-run", "--n-particles", "100"])
    assert result == 0


def test_cli_custom_box_size():
    """Test custom box size argument."""
    result = main(["--dry-run", "--box-size", "20.0"])
    assert result == 0


def test_cli_custom_dt():
    """Test custom timestep argument."""
    result = main(["--dry-run", "--dt", "0.001"])
    assert result == 0


def test_cli_all_custom_args():
    """Test all custom arguments together."""
    result = main(
        [
            "--dry-run",
            "--n-particles",
            "75",
            "--box-size",
            "15.0",
            "--dt",
            "0.005",
        ]
    )
    assert result == 0


def test_cli_no_args():
    """Test that no arguments shows help."""
    with pytest.raises(SystemExit):
        main([])


def test_cli_invalid_n_particles():
    """Test invalid n-particles type raises error."""
    with pytest.raises(SystemExit):
        main(["--dry-run", "--n-particles", "not_a_number"])


def test_cli_invalid_box_size():
    """Test invalid box-size type raises error."""
    with pytest.raises(SystemExit):
        main(["--dry-run", "--box-size", "not_a_number"])


def test_cli_invalid_dt():
    """Test invalid dt type raises error."""
    with pytest.raises(SystemExit):
        main(["--dry-run", "--dt", "not_a_number"])
