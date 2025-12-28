"""particles_simulation.system

Utilities to create particles and a cubic simulation box.
This module provides small helper classes used by tests and the
example CLI: Particle, Box and System (particle generation).
"""

from dataclasses import dataclass, field
from typing import Sequence, Union, List, Optional
import numpy as np
from numpy.random import Generator, default_rng


@dataclass
class Particle:
    """Simple container for particle state.

    The dataclass ensures concise semantics and easier comparisons in
    tests. Vector fields are normalized to numpy arrays of shape (3,).
    """
    position: np.ndarray = field(repr=True)
    velocity: np.ndarray = field(repr=True)
    acceleration: np.ndarray = field(repr=True)
    mass: float = field(repr=True)
    charge: float = field(repr=True)
    radius: float = field(repr=True)

    def __post_init__(self):
        # Convert sequence inputs to numpy arrays (shape checks for safety).
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)
        self.acceleration = np.asarray(self.acceleration, dtype=float)

        if self.position.shape != (3,) or self.velocity.shape != (3,) or self.acceleration.shape != (3,):
            raise ValueError("position, velocity and acceleration must be length-3 vectors")


class Box:
    """A possibly anisotropic rectangular box centered at the origin.

    `size` can be a scalar (cubic box) or a 3-sequence specifying edge
    lengths in x,y,z. All sizes must be positive.
    """

    def __init__(self, size:Union[float, Sequence[float]]):
        sizes = np.asarray(size, dtype=float)
        if sizes.size == 1:
            sizes = np.repeat(sizes.item(), 3)
        if sizes.shape != (3,):
            raise ValueError("size must be a scalar or a sequence of three positive numbers")
        if np.any(sizes <= 0):
            raise ValueError("box sizes must be positive")
        self.size = sizes

    def random_position(self, rng:Optional[Generator] = None) -> np.ndarray:
        """Return a random 3D position uniformly inside the box.

        If provided, use the supplied RNG for reproducibility.
        """
        rng = rng or default_rng()
        half = self.size / 2
        return rng.uniform(-half, half)

    def apply_reflective_boundary(self, particle: Particle) -> None:
        """Reflect a particle at the boundary, correcting out-of-bounds pos.

        If a particle lies outside the box on a given axis, set its
        coordinate to the boundary value and invert the corresponding
        velocity component so the particle is placed just inside the box.
        """
        half = self.size / 2
        for i in range(3):
            if particle.position[i] < -half[i]:
                particle.position[i] = -half[i]
                particle.velocity[i] = -particle.velocity[i]
            elif particle.position[i] > half[i]:
                particle.position[i] = half[i]
                particle.velocity[i] = -particle.velocity[i]


class System:
    """High-level system for sampling particle properties.

    This class centralizes sampling routines used to create collections of
    particles for tests and demos. A numpy Generator can be injected to
    make sampling deterministic and thread-safe for tests.
    """

    def __init__(self, box: Box, temperature:float = 300.0, rng:Optional[Generator] = None):
        self.box = box
        self.temperature = float(temperature)
        # Use provided RNG for reproducibility; fall back to a new Generator.
        self.rng: Generator = rng or default_rng()

    def round2(self, x):
        """Round numbers or arrays to two decimal places."""
        return np.round(x, 2)

    def sample_mass(self) -> float:
        """Sample a particle mass uniformly in [0.50, 1.50) and round it."""
        return float(self.round2(self.rng.uniform(0.50, 1.50)))

    def mass_to_radius(self, mass:float) -> float:
        """Convert mass to a radius value using a fixed scaling factor."""
        return float(self.round2(mass * 1.5))

    def sample_position(self) -> np.ndarray:
        """Sample and return a rounded random position inside the box."""
        return self.round2(self.box.random_position(rng=self.rng))

    def sample_mb(self, mass:float, T:float) -> np.ndarray:
        """Return a random instantaneous velocity-like vector (uniform).

        NOTE: Maxwell–Boltzmann sampling intentionally left as a future
        option; current approach samples uniformly in [-v_max, v_max]
        where v_max is 10% of the box size.
        """
        v_max = float(np.max(self.box.size) * 0.10)
        v = self.rng.uniform(-v_max, v_max, size=3)
        return self.round2(v)

    def create_particles(self, n_pos:int, n_neg:int, n_neutral:int) -> List[Particle]:
        """Create a list of particles with specified charge counts.

        Counts must be non-negative integers. Returns a list of Particle
        instances created with sampled properties.
        """
        for name, value in ("n_pos", n_pos), ("n_neg", n_neg), ("n_neutral", n_neutral):
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

        particles: List[Particle] = []

        def _make_particle(charge:float) -> Particle:
            mass = self.sample_mass()
            radius = self.mass_to_radius(mass)
            position = self.sample_position()
            velocity = self.sample_mb(mass, self.temperature)
            acceleration = self.sample_mb(mass, self.temperature)
            return Particle(position, velocity, acceleration, mass, charge, radius)

        for _ in range(n_pos):
            particles.append(_make_particle(charge=+1))
        for _ in range(n_neg):
            particles.append(_make_particle(charge=-1))
        for _ in range(n_neutral):
            particles.append(_make_particle(charge=0))

        return particles

