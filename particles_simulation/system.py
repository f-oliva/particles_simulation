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
    mass: float = field(repr=True)
    charge: float = field(repr=True)
    radius: float = field(repr=True)

    def __post_init__(self):
        # Convert sequence inputs to numpy arrays (shape checks for safety).
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)

        if self.position.shape != (3,) or self.velocity.shape != (3,):
            raise ValueError("position and velocity must be length-3 vectors")


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

    def check_and_correct(self, particle: Particle) -> np.ndarray:
        """Clamp a particle inside the box (accounting for its radius).

        Returns a boolean mask of shape (3,) indicating which axes had
        collisions and were corrected.
        """
        half = self.size / 2
        mask = np.zeros(3, dtype=bool)
        for i in range(3):
            if particle.position[i] < -half[i] + particle.radius:
                particle.position[i] = -half[i] + particle.radius
                mask[i] = True
            elif particle.position[i] > half[i] - particle.radius:
                particle.position[i] = half[i] - particle.radius
                mask[i] = True
        return mask

    def apply_reflective_boundary(self, particle: Particle) -> None:
        """Backward-compatible helper that corrects position and inverts velocity.

        This calls :meth:`check_and_correct` and flips the particle's
        *full-step* velocity components for axes where a correction occurred.
        """
        mask = self.check_and_correct(particle)
        if np.any(mask):
            particle.velocity[mask] = -particle.velocity[mask]


class System:
    """High-level system to create particle and properties.

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
        """Sample a particle mass uniformly in [0.20, 1.0) and round it."""
        return float(self.round2(self.rng.uniform(0.20, 1.0)))

    def mass_to_radius(self, mass:float) -> float:
        """Convert mass to a radius value using a fixed scaling factor."""
        return float(self.round2(mass * 0.5))

    def sample_position(self) -> np.ndarray:
        """Sample and return a rounded random position inside the box."""
        return self.round2(self.box.random_position(rng=self.rng))

    def sample_velocity(self) -> np.ndarray:
        """Return a random instantaneous velocity-like vector (uniform).

        NOTE: Maxwell–Boltzmann sampling intentionally left as a future
        option; current approach samples uniformly in [-v_max, v_max]
        where v_max is 10% of the box size.
        """
        v_max = float(np.max(self.box.size) * 0.50)
        v = self.rng.uniform(-v_max, v_max, size=3)
        return self.round2(v)

    def create_particle(self, charge:Optional[int] = None) -> Particle:
        """Create a single Particle using the System sampling routines.

        If `charge` is None, a charge is sampled randomly from {-1, 0, 1}
        using the system RNG to keep generation reproducible.
        """
        if charge is None:
            charge = int(self.rng.choice([-1, 0, 1]))
        mass = self.sample_mass()
        radius = self.mass_to_radius(mass)
        position = self.sample_position()
        velocity = self.sample_velocity()
        return Particle(position, velocity, mass, charge, radius)

    def create_particles(self, n_pos:int, n_neg:int, n_neutral:int) -> List[Particle]:
        """Create a list of particles with specified charge counts.

        Counts must be non-negative integers. Returns a list of Particle
        instances created with sampled properties.
        """
        for name, value in ("n_pos", n_pos), ("n_neg", n_neg), ("n_neutral", n_neutral):
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

        particles: List[Particle] = []

        for _ in range(n_pos):
            particles.append(self.create_particle(charge=+1))
        for _ in range(n_neg):
            particles.append(self.create_particle(charge=-1))
        for _ in range(n_neutral):
            particles.append(self.create_particle(charge=0))

        return particles

