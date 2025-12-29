"""Arcade-based interactive renderer for particles_simulation.

This module provides a simple, extensible Arcade app that can run the
simulator in a desktop window and supports basic interactive controls:
- Space: pause / resume
- A: add a particle at a random position
- R: remove the last particle
- Arrow keys: pan camera
- + / -: zoom in / out
- [ / ]: shrink / grow box by 10%

Note: This module is a *demo* and not used in automated tests that run in
headless CI. It aims to be a friendly starting point for interactive
experimentation with the integrator and system.
"""
from __future__ import annotations

import math
import random
from typing import Optional

import arcade
import numpy as np

from .system import System, Box, Particle
from .integrator import LeapFrogIntegrator

# Colors
CHARGE_COLOR = {1: arcade.color.RED, -1: arcade.color.BLUE, 0: arcade.color.GRAY}


class ArcadeRenderer(arcade.Window):
    """Arcade window that renders particles and lets the user interact.

    Parameters
    - integrator: LeapFrogIntegrator instance
    - width, height: window size in pixels
    - world_scale: pixels per world-unit
    """

    def __init__(self, integrator: LeapFrogIntegrator, width: int = 800, height: int = 600, title: str = "Particles Simulation", world_scale: float = 40.0):
        super().__init__(width, height, title)
        arcade.set_background_color(arcade.color.BLACK)

        self.integrator = integrator
        self.particles = integrator.particles
        self.box = getattr(integrator, "box", None)

        # Camera state (world units)
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.zoom = float(world_scale)

        self.paused = False

        # Control sensitivity
        self.pan_step = 0.5
        self.zoom_step = 1.1

        # Shoestring HUD font
        self.hud_font_size = 12

    # --- coordinate transforms -------------------------------------------------
    def world_to_screen(self, x: float, y: float) -> tuple[float, float]:
        """Convert world (x,y) to screen coordinates (pixels)."""
        cx = self.width / 2
        cy = self.height / 2
        sx = cx + (x - self.pan_x) * self.zoom
        sy = cy + (y - self.pan_y) * self.zoom
        return sx, sy

    # --- Arcade callbacks -----------------------------------------------------
    def on_draw(self):
        # Use Window.clear() inside on_draw; start_render() is for module-level
        # usage and can only be called once in an application's lifetime.
        self.clear()

        # Draw box boundaries (as rectangle)
        if self.box is not None:
            half = self.box.size / 2
            # four corners in world coords
            x0, y0 = self.world_to_screen(-half[0], -half[1])
            x1, y1 = self.world_to_screen(half[0], half[1])
            # arcade's API may vary; try the lrbt helper and fall back to a
            # centered rectangle outline which is broadly available.
            try:
                arcade.draw_lrbt_rectangle_outline(x0, x1, y0, y1, arcade.color.WHITE)
            except AttributeError:
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                w = abs(x1 - x0)
                h = abs(y1 - y0)
                arcade.draw_rectangle_outline(cx, cy, w, h, arcade.color.WHITE)

        # Draw particles
        for p in self.particles:
            sx, sy = self.world_to_screen(p.position[0], p.position[1])
            r = max(1.0, p.radius * self.zoom)
            color = CHARGE_COLOR.get(int(p.charge), arcade.color.WHITE)
            arcade.draw_circle_filled(sx, sy, r, color)

        # HUD
        paused_text = "PAUSED" if self.paused else "RUNNING"
        arcade.draw_text(f"{paused_text}  Particles: {len(self.particles)}  Zoom: {self.zoom:.1f}", 10, 10, arcade.color.WHITE, self.hud_font_size)

    def on_update(self, delta_time: float):
        # Advance simulation by one step per update when running
        if not self.paused:
            # calling step(1) to keep physics and rendering in sync
            self.integrator.step(1)

    def on_key_press(self, key, modifiers):
        # Pause/resume
        if key == arcade.key.SPACE:
            self.paused = not self.paused
        # Add a particle at a random position inside the box
        elif key == arcade.key.A:
            self._add_particle()
        elif key == arcade.key.R:
            if self.particles:
                self.particles.pop()
        elif key == arcade.key.UP:
            self.pan_y -= self.pan_step
        elif key == arcade.key.DOWN:
            self.pan_y += self.pan_step
        elif key == arcade.key.LEFT:
            self.pan_x -= self.pan_step
        elif key == arcade.key.RIGHT:
            self.pan_x += self.pan_step
        elif key == arcade.key.EQUAL or key == arcade.key.PLUS:
            self.zoom *= self.zoom_step
        elif key == arcade.key.MINUS:
            self.zoom /= self.zoom_step
        elif key == arcade.key.LBRACKET:
            self._change_box_scale(0.9)
        elif key == arcade.key.RBRACKET:
            self._change_box_scale(1.1)

    # --- helpers ---------------------------------------------------------------
    def _add_particle(self):
        if self.box is None:
            return
        pos = self.box.random_position(rng=None)
        v = np.zeros(3)
        a = np.zeros(3)
        mass = 1.0
        charge = random.choice([-1, 0, 1])
        radius = 0.1
        p = Particle(position=pos, velocity=v, acceleration=a, mass=mass, charge=charge, radius=radius)
        self.particles.append(p)
        # Keep integrator internal lists consistent if needed
        if hasattr(self.integrator, "v_half"):
            self.integrator.v_half.append(np.asarray(p.velocity) + 0.5 * np.asarray(p.acceleration) * self.integrator.dt)

    def _change_box_scale(self, factor: float):
        if self.box is None:
            return
        new_size = self.box.size * factor
        # clamp to reasonable ranges
        min_size = 1e-2
        max_size = 1e3
        new_size = np.clip(new_size, min_size, max_size)
        self.box.size = new_size

        # ensure particles are inside the new box
        half = self.box.size / 2
        for p in self.particles:
            for i in range(3):
                if p.position[i] < -half[i]:
                    p.position[i] = -half[i]
                elif p.position[i] > half[i]:
                    p.position[i] = half[i]


def run_arcade_demo(n_particles: int = 50, box_size: float = 10.0, dt: float = 0.01):
    """Convenience function to run a demo app using Arcade.

    This creates a `System`, a `LeapFrogIntegrator`, populates initial
    particles, and runs the Arcade app. Close the window to exit.
    """
    box = Box(box_size)
    s = System(box, rng=None)
    ps = s.create_particles(n_pos=n_particles // 3, n_neg=n_particles // 3, n_neutral=n_particles - 2*(n_particles // 3))

    integ = LeapFrogIntegrator(box, ps, dt=dt)

    app = ArcadeRenderer(integ)
    arcade.run()


if __name__ == "__main__":
    run_arcade_demo()
