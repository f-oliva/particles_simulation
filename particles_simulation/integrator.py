"""Simple leap-frog integrator for the particles_simulation package.

This module provides a lightweight integrator that evolves a collection of
`Particle` objects inside a `Box` using a leap-frog scheme. For now inter-
particle forces are ignored by default; a custom `force_func` may be
provided to compute accelerations (force/mass) given a particle and the
full particle list.
"""
from typing import Callable, List, Optional
import numpy as np

from .system import Box, Particle

ForceFunc = Callable[[Particle, List[Particle]], np.ndarray]


class LeapFrogIntegrator:
    """Leap-frog integrator.

    The integrator stores the half-step velocities internally and keeps the
    Particle objects' `position` and `velocity` fields up to date (velocity is
    kept as the full-step velocity for easier inspection/tests).

    Parameters
    - box: simulation box used for reflective boundaries
    - particles: list of Particle instances to integrate
    - dt: timestep (float)
    - force_func: function(particle, particles) -> acceleration vector; if
      omitted, the integrator uses `particle.acceleration` as a constant
      acceleration (useful while interactions are not implemented).
    """

    def __init__(self, box: Box, particles: List[Particle], dt: float, force_func: Optional[ForceFunc] = None):
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.box = box
        self.particles = list(particles)
        self.dt = float(dt)
        self.force_func: ForceFunc = force_func or (lambda p, ps: p.acceleration)

        # Initialize half-step velocities: v_{n+1/2} = v_n + 0.5 * a_n * dt
        self.v_half: List[np.ndarray] = [np.asarray(p.velocity, dtype=float) + 0.5 * np.asarray(p.acceleration, dtype=float) * self.dt for p in self.particles]

    def step(self, n_steps: int = 1) -> None:
        """Advance the simulation `n_steps` steps using leap-frog.

        Boundaries are reflective: when a particle crosses an axis boundary it is
        placed exactly at the boundary and the corresponding velocity component
        (half-step velocity) is inverted.
        """
        for _ in range(int(n_steps)):
            for i, p in enumerate(self.particles):
                vh = self.v_half[i]
                # store current half velocity for computing full-step velocity later
                vh_old = vh.copy()

                # Drift: advance position by v_{n+1/2} * dt
                p.position = p.position + vh_old * self.dt

                # Reflective boundaries handled here so we update v_half consistently
                half = self.box.size / 2
                for ax in range(3):
                    if p.position[ax] < -half[ax]:
                        p.position[ax] = -half[ax]
                        vh_old[ax] = -vh_old[ax]
                    elif p.position[ax] > half[ax]:
                        p.position[ax] = half[ax]
                        vh_old[ax] = -vh_old[ax]

                # Compute new acceleration (force/mass). By default this returns the
                # particle's own `acceleration` attribute which allows a simple
                # constant-acceleration test setup.
                a_new = np.asarray(self.force_func(p, self.particles), dtype=float)

                # Kick: update half-step velocity to v_{n+3/2} = v_{n+1/2} + a_{n+1} * dt
                vh_new = vh_old + a_new * self.dt

                # Update full-step velocity for external visibility: v_{n+1} = v_{n+1/2} + 0.5 * a_{n+1} * dt
                p.velocity = vh_old + 0.5 * a_new * self.dt

                # Store updated half-step velocity
                self.v_half[i] = vh_new

    def positions(self) -> np.ndarray:
        """Return an array of particle positions (N, 3)."""
        return np.vstack([p.position for p in self.particles])

    def velocities(self) -> np.ndarray:
        """Return an array of particle full-step velocities (N, 3)."""
        return np.vstack([p.velocity for p in self.particles])
