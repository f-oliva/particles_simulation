import numpy as np
import pytest
from numpy.random import default_rng

from particles_simulation.system import Box, Particle, System


def test_sample_mass_range():
    rng = default_rng(123)
    sys = System(Box(10.0), rng=rng)
    mass = sys.sample_mass()
    assert 0.5 <= mass < 1.5


def test_reproducible_sampling():
    rng1 = default_rng(42)
    rng2 = default_rng(42)
    s1 = System(Box(10.0), rng=rng1)
    s2 = System(Box(10.0), rng=rng2)

    p1 = s1.create_particles(1, 0, 0)[0]
    p2 = s2.create_particles(1, 0, 0)[0]

    assert np.allclose(p1.position, p2.position)
    assert np.allclose(p1.velocity, p2.velocity)
    assert p1.mass == p2.mass


def test_reflective_boundary_corrects_position():
    box = Box(2.0)  # half-size = 1.0
    # place particle outside on +x side with radius 1.0
    particle = Particle(
        position=[2.0, 0.0, 0.0],
        velocity=[1.0, 0.0, 0.0],
        mass=1.0,
        charge=0,
        radius=1.0,
    )
    box.apply_reflective_boundary(particle)
    # corrected position should be at half - radius = 1.0 - 1.0 = 0.0
    assert pytest.approx(particle.position[0]) == 0.0
    assert pytest.approx(particle.velocity[0]) == -1.0


def test_box_size_validation():
    with pytest.raises(ValueError):
        Box(0)
    with pytest.raises(ValueError):
        Box([-1, 1, 1])
    # valid anisotropic box
    b = Box([1, 2, 3])
    assert np.allclose(b.size, np.array([1.0, 2.0, 3.0]))


def test_create_particles_counts_and_types():
    rng = default_rng(7)
    s = System(Box(10.0), rng=rng)
    ps = s.create_particles(2, 1, 1)
    assert len(ps) == 4
    charges = sorted([p.charge for p in ps])
    assert charges == [-1, 0, 1, 1]
    assert all(isinstance(p, Particle) for p in ps)


def test_create_particle_random_charge_reproducible():
    rng = default_rng(123)
    s1 = System(Box(10.0), rng=rng)
    charges1 = [s1.create_particle().charge for _ in range(5)]

    rng2 = default_rng(123)
    s2 = System(Box(10.0), rng=rng2)
    charges2 = [s2.create_particle().charge for _ in range(5)]

    assert charges1 == charges2
    assert all(c in (-1, 0, 1) for c in charges1)
