import numpy as np
import pytest
from numpy.random import default_rng

from particles_simulation.system import Box, Particle, System
from particles_simulation.integrator import LeapFrogIntegrator


def test_leapfrog_updates_position_no_acc():
    box = Box(10.0)
    p = Particle(position=[0.0, 0.0, 0.0], velocity=[1.0, 0.0, 0.0], acceleration=[0.0, 0.0, 0.0], mass=1.0, charge=0, radius=0.1)
    integ = LeapFrogIntegrator(box, [p], dt=0.1)

    integ.step()

    # v_half initial = 1.0 -> position moves by 0.1
    assert np.allclose(p.position, np.array([0.1, 0.0, 0.0]))
    # with zero acceleration velocity remains 1.0
    assert np.allclose(p.velocity, np.array([1.0, 0.0, 0.0]))


def test_leapfrog_bounces():
    box = Box(2.0)  # half-size = 1.0
    # start near +x boundary moving outward
    p = Particle(position=[0.95, 0.0, 0.0], velocity=[1.0, 0.0, 0.0], acceleration=[0.0, 0.0, 0.0], mass=1.0, charge=0, radius=0.1)
    integ = LeapFrogIntegrator(box, [p], dt=0.1)

    integ.step()

    # particle should be placed at the boundary and have inverted velocity
    assert pytest.approx(p.position[0]) == 1.0
    assert pytest.approx(p.velocity[0]) == -1.0


def test_reproducible_with_rng_and_multiple_particles():
    rng = default_rng(123)
    box = Box(10.0)
    # create two particles using deterministic RNG
    p1 = Particle(position=box.random_position(rng), velocity=[0, 0, 0], acceleration=[0, 0, 0], mass=1.0, charge=0, radius=0.1)
    p2 = Particle(position=box.random_position(rng), velocity=[0, 0, 0], acceleration=[0, 0, 0], mass=1.0, charge=0, radius=0.1)
    integ = LeapFrogIntegrator(box, [p1, p2], dt=0.05)

    integ.step(10)

    # basic sanity: positions are finite and shape matches
    pos = integ.positions()
    assert pos.shape == (2, 3)
    assert np.all(np.isfinite(pos))


def test_system_six_particles_evolve_20_steps():
    rng = default_rng(2025)
    box = Box(10.0)
    s = System(box, rng=rng)
    ps = s.create_particles(2, 2, 2)
    assert len(ps) == 6

    initial_positions = np.vstack([p.position.copy() for p in ps])
    initial_charges = [p.charge for p in ps]

    integ = LeapFrogIntegrator(box, ps, dt=0.02)
    integ.step(20)

    pos = integ.positions()
    vel = integ.velocities()

    assert pos.shape == (6, 3)
    assert np.all(np.isfinite(pos)) and np.all(np.isfinite(vel))

    half = box.size / 2
    # all particles should remain inside the box after integration
    assert np.all(np.abs(pos) <= half + 1e-12)

    # charges preserved and at least some motion occurred
    assert sorted(initial_charges) == sorted([p.charge for p in ps])
    assert not np.allclose(initial_positions, pos)


def test_reproducible_evolution_six_particles():
    rng1 = default_rng(2025)
    rng2 = default_rng(2025)
    box1 = Box(10.0)
    box2 = Box(10.0)

    s1 = System(box1, rng=rng1)
    s2 = System(box2, rng=rng2)

    ps1 = s1.create_particles(2, 2, 2)
    ps2 = s2.create_particles(2, 2, 2)

    integ1 = LeapFrogIntegrator(box1, ps1, dt=0.02)
    integ2 = LeapFrogIntegrator(box2, ps2, dt=0.02)

    integ1.step(20)
    integ2.step(20)

    assert np.allclose(integ1.positions(), integ2.positions())
    assert np.allclose(integ1.velocities(), integ2.velocities())
