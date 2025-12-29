"""Command-line interface for the particles_simulation package.

This lightweight CLI provides a small entry-point to run demos such as
an Arcade-based interactive renderer. The CLI is intentionally minimal
and should be extended as needed.
"""
from __future__ import annotations

from typing import List, Optional
import argparse


def main(argv: Optional[List[str]] = None) -> int:
    """Parse command-line arguments and execute the requested action.

    Parameters
    - argv: list of arguments (defaults to sys.argv[1:]). This allows tests
      to call `main` without spawning subprocesses.

    Returns an exit code (0 on success).
    """
    parser = argparse.ArgumentParser(prog="particles_simulation")

#    parser.add_argument("--version", action="version", version="particles_simulation")

    parser.add_argument("--arcade-demo", action="store_true", help="Run the Arcade-based interactive demo", default=True)
    parser.add_argument("--n-particles", type=int, default=50, help="Number of particles for the demo")
    parser.add_argument("--box-size", type=float, default=10.0, help="Initial box size for the demo")
    parser.add_argument("--dt", type=float, default=0.01, help="Integrator timestep for the demo")
    parser.add_argument("--dry-run", action="store_true", help="Print demo parameters and exit without launching GUI (useful for testing)", default=False, required=False)

    args = parser.parse_args(argv)

    if args.arcade_demo:
        # Support dry-run without importing Arcade so tests and environments
        # without Arcade can still verify CLI behavior.
        if args.dry_run:
            print("Arcade demo dry-run: parameters")
            print(f"  n_particles: {args.n_particles}")
            print(f"  box_size:    {args.box_size}")
            print(f"  dt:          {args.dt}")
            return 0

        # Import the module lazily to avoid requiring Arcade for other CLI actions
        try:
            from .arcade_renderer import run_arcade_demo
        except Exception as exc:  # ImportError or other import-time error
            print("Arcade demo is unavailable: could not import Arcade or renderer module.")
            print(f"Reason: {exc}")
            return 2

        # Launch the demo (blocking call which opens a native window)
        run_arcade_demo(n_particles=args.n_particles, box_size=args.box_size, dt=args.dt)
        return 0

    # Default behavior: print help
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
