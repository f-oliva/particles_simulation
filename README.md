# particles_simulation

[![codecov](https://codecov.io/gh/f-oliva/particles_simulation/branch/main/graph/badge.svg?token=particles_simulation_token_here)](https://codecov.io/gh/f-oliva/particles_simulation)
[![CI](https://github.com/f-oliva/particles_simulation/actions/workflows/main.yml/badge.svg)](https://github.com/f-oliva/particles_simulation/actions/workflows/main.yml)

A lightweight particle simulation library with an interactive Arcade renderer. Simulate particles in a bounded box with a leap-frog integrator, with optional support for custom force functions.

## Features

- **Leap-frog integrator**: Symplectic time integration for accurate particle dynamics
- **Particle system**: Create and manage particles with position, velocity, mass, charge, and radius
- **Reflective boundaries**: Particles bounce off box walls
- **Arcade renderer**: Interactive visualization with keyboard controls
- **Extensible**: Plug in custom force functions for inter-particle interactions

## Installation

Install from source:

```bash
git clone https://github.com/f-oliva/particles_simulation
cd particles_simulation
pip install -e .
```

## Quick Start

### Interactive Demo

Run the interactive Arcade-based demo:

```bash
python -m particles_simulation
# or
particles_simulation
```

Options:
- `--n-particles`: Number of particles (default: 50)
- `--box-size`: Initial box size (default: 10.0)
- `--dt`: Integrator timestep (default: 0.01)

### Programmatic Usage

```python
from particles_simulation.system import Particle, Box, System
from particles_simulation.integrator import LeapFrogIntegrator

# Create a box and particles
box = Box(size=10.0)
system = System(box, n_particles=50)
particles = system.particles

# Create an integrator
integrator = LeapFrogIntegrator(
    box=box,
    particles=particles,
    dt=0.01
)

# Advance the simulation
integrator.step(n_steps=100)

# Access particle state
for particle in integrator.particles:
    print(f"Position: {particle.position}, Velocity: {particle.velocity}")
```

### With Custom Forces

```python
import numpy as np

def your_force(particle, all_particles):
    """Custom force function for particles.
    
    Args:
        particle: Particle to compute force for
        all_particles: List of all particles in the simulation
        
    Returns:
        np.ndarray: Acceleration vector (force/mass)
    """
    # Implement your force calculation here
    return np.zeros(3)

integrator = LeapFrogIntegrator(
    box=box,
    particles=particles,
    dt=0.01,
    force_func=your_force
)
```

## Arcade Controls

When running the interactive demo:

- **Space**: Pause / resume
- **A**: Add a random particle
- **R**: Remove the last particle
- **Arrow keys**: Pan camera
- **+/-**: Zoom in / out
- **[/]**: Shrink / grow box by 10%

## Development

Install the project in development mode:

```bash
pip install -e ".[test]"
```

Run tests:

```bash
pytest
```

Format and lint code with ruff:

```bash
# Format code
ruff format src/ 

# Lint code
ruff check src/

# Lint and fix issues automatically
ruff check --fix src/
```

Type checking with mypy:

```bash
mypy src/particles_simulation/
```


