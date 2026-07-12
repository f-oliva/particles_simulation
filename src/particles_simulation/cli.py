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
    parser = argparse.ArgumentParser(prog="particles-sim")
    subparsers = parser.add_subparsers(dest="command")

    demo_parser = subparsers.add_parser("demo", help="Run the Arcade demo")
    demo_parser.add_argument("--n-particles", type=int, default=50, help="Number of particles for the demo")
    demo_parser.add_argument("--box-size", type=float, default=10.0, help="Initial box size for the demo")
    demo_parser.add_argument("--dt", type=float, default=0.01, help="Integrator timestep for the demo")
    demo_parser.add_argument("--dry-run", action="store_true", help="Print demo parameters and exit without launching GUI (useful for testing)", default=False)

    args = parser.parse_args(argv)

    if args.command == "demo":
        if args.dry_run:
            print("Arcade demo dry-run: parameters")
            print(f"  n_particles: {args.n_particles}")
            print(f"  box_size:    {args.box_size}")
            print(f"  dt:          {args.dt}")
            return 0

        try:
            from .arcade_renderer import run_arcade_demo
        except (ImportError, ModuleNotFoundError) as exc:
            print("Arcade demo is unavailable: could not import Arcade or renderer module.")
            print(f"Reason: {exc}")
            return 2

        run_arcade_demo(n_particles=args.n_particles, box_size=args.box_size, dt=args.dt)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
