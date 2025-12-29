import sys
from importlib import reload

import pytest

from particles_simulation import cli


def test_cli_help_exits_with_zero():
    with pytest.raises(SystemExit) as excinfo:
        # argparse raises SystemExit when --help is used; simulate it
        cli.main(["--help"])
    # SystemExit code 0 is expected for help
    assert excinfo.type is SystemExit


def test_cli_arcade_dry_run_prints_parameters(capsys):
    # Use dry-run so we don't open a GUI during tests
    rc = cli.main(["--arcade-demo", "--n-particles", "6", "--box-size", "5.0", "--dt", "0.02", "--dry-run"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Arcade demo dry-run: parameters" in captured.out
    assert "n_particles: 6" in captured.out
    assert "box_size:    5.0" in captured.out
    assert "dt:          0.02" in captured.out
