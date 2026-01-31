"""Tests for the leap-frog integrator."""

import numpy as np
import pytest
from numpy.random import default_rng

from particles_simulation.system import Box, System, Particle
from particles_simulation.integrator import LeapFrogIntegrator


def test_integrator_initialization():
    """Test integrator initializes correctly."""
    box = Box(10.0)
    particles = [Particle([0, 0, 0], [1, 0, 0], 1.0, 0, 0.5)]
    integrator = LeapFrogIntegrator(box, particles, dt=0.01)

    assert integrator.dt == 0.01
    assert len(integrator.particles) == 1
    assert len(integrator.v_half) == 1


def test_integrator_dt_validation():
    """Test that integrator validates positive dt."""
    box = Box(10.0)
    particles = [Particle([0, 0, 0], [1, 0, 0], 1.0, 0, 0.5)]

    with pytest.raises(ValueError, match="dt must be positive"):
        LeapFrogIntegrator(box, particles, dt=0)

    with pytest.raises(ValueError, match="dt must be positive"):
        LeapFrogIntegrator(box, particles, dt=-0.1)


def test_integrator_single_step_free_particle():
    """Test single integration step with free particle (no forces)."""
    box = Box(100.0)  # Large box to avoid boundaries
    initial_pos = np.array([0.0, 0.0, 0.0])
    initial_vel = np.array([1.0, 0.0, 0.0])
    particle = Particle(initial_pos.copy(), initial_vel.copy(), 1.0, 0, 0.5)
    particles = [particle]

    integrator = LeapFrogIntegrator(box, particles, dt=0.01)
    integrator.step(1)

    # With no forces, particle should move according to initial velocity
    expected_new_pos = initial_pos + initial_vel * 0.01
    assert np.allclose(particle.position, expected_new_pos, atol=1e-10)
    # Velocity should remain unchanged
    assert np.allclose(particle.velocity, initial_vel, atol=1e-10)


def test_integrator_multiple_steps():
    """Test multiple integration steps accumulate correctly."""
    box = Box(100.0)
    particle = Particle([0, 0, 0], [1, 0, 0], 1.0, 0, 0.5)
    particles = [particle]

    integrator = LeapFrogIntegrator(box, particles, dt=0.01)
    integrator.step(10)

    # After 10 steps with constant velocity, position should be 0.1 in x
    expected_x = 0.1
    assert np.allclose(particle.position[0], expected_x, atol=1e-10)
    assert np.allclose(particle.position[1], 0.0, atol=1e-10)
    assert np.allclose(particle.position[2], 0.0, atol=1e-10)


def test_integrator_positions_velocities():
    """Test positions() and velocities() return correct arrays."""
    box = Box(100.0)
    p1 = Particle([1, 0, 0], [1, 0, 0], 1.0, 0, 0.5)
    p2 = Particle([0, 1, 0], [0, 1, 0], 1.0, 0, 0.5)
    particles = [p1, p2]

    integrator = LeapFrogIntegrator(box, particles, dt=0.01)

    positions = integrator.positions()
    velocities = integrator.velocities()

    assert positions.shape == (2, 3)
    assert velocities.shape == (2, 3)
    assert np.allclose(positions[0], [1, 0, 0])
    assert np.allclose(positions[1], [0, 1, 0])
    assert np.allclose(velocities[0], [1, 0, 0])
    assert np.allclose(velocities[1], [0, 1, 0])


def test_integrator_with_constant_force():
    """Test integrator with constant acceleration (constant force)."""
    box = Box(100.0)
    particle = Particle([0, 0, 0], [0, 0, 0], 1.0, 0, 0.5)
    particles = [particle]

    # Constant acceleration in x direction: a = 1.0
    def constant_force(p, all_particles):
        return np.array([1.0, 0.0, 0.0])

    integrator = LeapFrogIntegrator(box, particles, dt=0.01, force_func=constant_force)

    # After 1 step with a=1.0 and dt=0.01:
    # v_{1} = 0 + 0.5 * 1.0 * 0.01 = 0.005 (at step end)
    # x_{1} = 0 + 0.005 * 0.01 = 0.00005
    integrator.step(1)

    # Position should increase (acceleration acts)
    assert particle.position[0] > 0
    # Velocity should increase (acceleration acts)
    assert particle.velocity[0] > 0


def test_integrator_reflective_boundary():
    """Test that particles bounce off boundaries correctly."""
    box = Box(2.0)  # half-size = 1.0
    # Particle moving toward +x boundary
    particle = Particle([0.9, 0, 0], [1.0, 0, 0], 1.0, 0, 0.1)
    particles = [particle]

    integrator = LeapFrogIntegrator(box, particles, dt=0.5)
    integrator.step(1)

    # Particle should be near boundary and velocity should be reversed
    assert particle.position[0] <= 1.0 - 0.1  # Within bounds accounting for radius
    assert particle.velocity[0] < 0  # Bounced back


def test_integrator_zero_particles():
    """Test integrator handles zero particles gracefully."""
    box = Box(10.0)
    integrator = LeapFrogIntegrator(box, [], dt=0.01)

    # Should not raise error
    integrator.step(10)
    assert len(integrator.particles) == 0


def test_integrator_energy_conservation_no_forces():
    """Test energy is conserved in free particle motion (no boundaries)."""
    box = Box(1000.0)  # Very large to avoid boundaries
    particle = Particle([0, 0, 0], [1, 1, 1], 2.0, 0, 0.5)
    particles = [particle]

    integrator = LeapFrogIntegrator(box, particles, dt=0.01)

    # Kinetic energy should be conserved
    initial_ke = 0.5 * 2.0 * np.sum(particle.velocity ** 2)

    integrator.step(100)

    final_ke = 0.5 * 2.0 * np.sum(particle.velocity ** 2)
    assert np.allclose(initial_ke, final_ke, rtol=1e-10)
