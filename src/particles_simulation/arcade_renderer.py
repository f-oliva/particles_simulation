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
from typing import Optional, List, Dict

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

    def __init__(self, integrator: LeapFrogIntegrator, width: int = 1200, height: int = 800, title: str = "Particles Simulation", world_scale: float = 40.0):
        super().__init__(width, height, title)
        arcade.set_background_color(arcade.color.BLACK)

        self.integrator = integrator
        self.particles = integrator.particles
        self.box = getattr(integrator, "box", None)
        # Create a System instance for sampling particle properties when a box is available
        self.system = System(self.box) if self.box is not None else None

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

        # 3D rotation state (radians) and mouse state
        self.rot_x = 0.0  # pitch
        self.rot_y = 0.0  # yaw
        self.rot_z = 0.0  # roll
        self._rotating_primary = False   # left button -> pitch/yaw
        self._rotating_secondary = False # right button -> roll
        self._last_mouse_x = 0.0
        self._last_mouse_y = 0.0
        # Rotation sensitivity: radians per pixel moved
        self.rotation_sensitivity = 0.01
        # Camera distance for perspective projection
        self.camera_distance = float(np.max(self.box.size)) * 2.0 if self.box is not None else 10.0
        self.min_camera_distance = 0.5
        self.max_camera_distance = 1e6

        # UI buttons (simple clickable rectangles)
        # Each button: {'label': str, 'rect': (x, y, w, h), 'callback': callable}
        self._buttons: List[Dict] = []
        self._init_buttons()

        # keep a snapshot of initial particles for restart
        self._initial_particles = [Particle(position=p.position.copy(), velocity=p.velocity.copy(), mass=p.mass, charge=p.charge, radius=p.radius) for p in self.particles]

    # --- coordinate transforms -------------------------------------------------
    def world_to_screen(self, x: float, y: float, z: float = 0.0) -> tuple[float, float, float, float]:
        """Project a 3D world point to screen coordinates with simple perspective.

        Returns (sx, sy, factor, zcam) where `factor` is the perspective scale applied
        to distances (useful to scale radii) and `zcam` is the camera-space z value
        (larger = closer to camera), used for depth-sorting.
        """
        cx = self.width / 2
        cy = self.height / 2

        # Translate relative to camera pan (we treat pan as world-space x/y offset)
        vx = x - self.pan_x
        vy = y - self.pan_y
        vz = z

        # Apply rotations: yaw (y), pitch (x), roll (z)
        cyaw = math.cos(self.rot_y)
        syaw = math.sin(self.rot_y)
        cp = math.cos(self.rot_x)
        sp = math.sin(self.rot_x)
        cr = math.cos(self.rot_z)
        sr = math.sin(self.rot_z)

        # Rotation around Y (yaw)
        x1 = vx * cyaw + vz * syaw
        z1 = -vx * syaw + vz * cyaw
        # Rotation around X (pitch)
        y2 = vy * cp - z1 * sp
        z2 = vy * sp + z1 * cp
        # Rotation around Z (roll)
        x3 = x1 * cr - y2 * sr
        y3 = x1 * sr + y2 * cr
        z3 = z2

        # Perspective projection: camera at z = +camera_distance looking toward origin
        denom = (self.camera_distance - z3)
        if denom <= 1e-6:
            factor = 1e6
        else:
            factor = self.camera_distance / denom

        sx = cx + x3 * factor * self.zoom
        sy = cy + y3 * factor * self.zoom
        # camera-space depth is z3 (after rotations)
        zcam = z3
        return sx, sy, factor, zcam

    # --- Arcade callbacks -----------------------------------------------------
    def on_draw(self):
        # Use Window.clear() inside on_draw; start_render() is for module-level
        # usage and can only be called once in an application's lifetime.
        self.clear()

        # Draw box boundaries (as rectangle)
        # Project box corners and particles and depth-sort draw operations so nearer
        # items occlude farther ones.
        draw_items: list[tuple[float, str, tuple]] = []  # (depth, kind, data)

        if self.box is not None:
            half = self.box.size / 2
            # eight corners of the box (3D)
            corners = [
                (-half[0], -half[1], -half[2]), (half[0], -half[1], -half[2]),
                (half[0], half[1], -half[2]), (-half[0], half[1], -half[2]),
                (-half[0], -half[1], half[2]), (half[0], -half[1], half[2]),
                (half[0], half[1], half[2]), (-half[0], half[1], half[2])
            ]
            pts = [self.world_to_screen(x, y, z) for x, y, z in corners]
            # get z (camera-space depth) for each corner
            zvals = [z for (_, _, _, z) in pts]
            pts_xy = [(sx, sy) for (sx, sy, _, _) in pts]

            # edges to draw
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # back face
                (4, 5), (5, 6), (6, 7), (7, 4),  # front face
                (0, 4), (1, 5), (2, 6), (3, 7)   # side edges
            ]
            for a, b in edges:
                x0, y0 = pts_xy[a]
                x1, y1 = pts_xy[b]
                depth = (zvals[a] + zvals[b]) / 2.0
                draw_items.append((depth, "edge", (x0, y0, x1, y1)))

        # Particles
        for p in self.particles:
            sx, sy, factor, zcam = self.world_to_screen(p.position[0], p.position[1], p.position[2])
            r = max(1.0, p.radius * self.zoom * factor)
            color = CHARGE_COLOR.get(int(p.charge), arcade.color.WHITE)
            draw_items.append((zcam, "particle", (sx, sy, r, color)))

        # Sort by depth (small/negative z -> far, large z -> near); draw farthest first
        draw_items.sort(key=lambda it: it[0])
        for _, kind, data in draw_items:
            if kind == "edge":
                x0, y0, x1, y1 = data
                arcade.draw_line(x0, y0, x1, y1, arcade.color.WHITE)
            else:  # particle
                sx, sy, r, color = data
                arcade.draw_circle_filled(sx, sy, r, color)

        # HUD
        paused_text = "PAUSED" if self.paused else "RUNNING"
        rot_x_deg = math.degrees(self.rot_x)
        rot_y_deg = math.degrees(self.rot_y)
        rot_z_deg = math.degrees(self.rot_z)
        arcade.draw_text(
            f"{paused_text}  Particles: {len(self.particles)}  Zoom: {self.zoom:.1f}  Rot(X/Y/Z): {rot_x_deg:.0f}/{rot_y_deg:.0f}/{rot_z_deg:.0f}°  CamDist: {self.camera_distance:.1f}",
            10, 40, arcade.color.WHITE, self.hud_font_size
        )

        # Draw UI buttons
        self._draw_buttons()
    def on_mouse_press(self, x: float, y: float, button: int, modifiers: int):
        # Check UI buttons first (clicks on buttons should not start rotations)
        if button == arcade.MOUSE_BUTTON_LEFT and self._handle_button_click(x, y):
            return

        if button == arcade.MOUSE_BUTTON_LEFT:
            self._rotating_primary = True
            self._last_mouse_x = x
            self._last_mouse_y = y
        elif button == arcade.MOUSE_BUTTON_RIGHT:
            self._rotating_secondary = True
            self._last_mouse_x = x
            self._last_mouse_y = y

    def on_mouse_release(self, x: float, y: float, button: int, modifiers: int):
        if button == arcade.MOUSE_BUTTON_LEFT:
            self._rotating_primary = False
        elif button == arcade.MOUSE_BUTTON_RIGHT:
            self._rotating_secondary = False

    # --- Buttons & UI helpers -------------------------------------------------
    def _init_buttons(self):
        padding = 8
        btn_w = 120
        btn_h = 28
        x = 10 + btn_w / 2
        # Start near top-left under HUD text and stack vertically
        y_start = self.height - 60
        labels = ["Add Particle", "Remove Particle", "Pause Simulation", "Restart Simulation"]
        callbacks = [self._btn_add, self._btn_remove, self._btn_pause_toggle, self._btn_restart]
        self._buttons = []
        for i, (label, cb) in enumerate(zip(labels, callbacks)):
            y = y_start - i * (btn_h + padding)
            self._buttons.append({"label": label, "rect": (x, y, btn_w, btn_h), "callback": cb})

    def _draw_buttons(self):
        for b in self._buttons:
            x, y, w, h = b["rect"]
            # convert center-based (x,y) to left-bottom (lx, ly)
            lx = x - w / 2
            ly = y - h / 2
            pts = [(lx, ly), (lx + w, ly), (lx + w, ly + h), (lx, ly + h)]
            arcade.draw_polygon_filled(pts, arcade.color.DARK_GRAY)
            arcade.draw_polygon_outline(pts, arcade.color.WHITE)
            # text positioned with small padding from left and vertically centered
            text_x = lx + 6
            text_y = ly + h / 2 - 6
            arcade.draw_text(b["label"], text_x, text_y, arcade.color.WHITE, 12)

    def _handle_button_click(self, sx: float, sy: float) -> bool:
        # arcade's origin is bottom-left; our rects are centered at (x,y)
        for b in self._buttons:
            x, y, w, h = b["rect"]
            if (sx >= x - w / 2 and sx <= x + w / 2 and sy >= y - h / 2 and sy <= y + h / 2):
                b["callback"]()
                return True
        return False

    # --- button callbacks ----------------------------------------------------
    def _btn_add(self):
        self._add_particle()

    def _btn_remove(self):
        if self.particles:
            self.particles.pop()
            if hasattr(self.integrator, "v_half") and self.integrator.v_half:
                self.integrator.v_half.pop()

    def _btn_pause_toggle(self):
        self.paused = not self.paused
        # update label
        for b in self._buttons:
            if b["callback"] is self._btn_pause_toggle:
                b["label"] = "Resume" if self.paused else "Pause"

    def _btn_restart(self):
        # Restore initial snapshot
        self.particles[:] = [Particle(position=p.position.copy(), velocity=p.velocity.copy(), mass=p.mass, charge=p.charge, radius=p.radius) for p in self._initial_particles]
        # reset integrator state
        self.integrator.particles = self.particles
        if hasattr(self.integrator, "v_half"):
            self.integrator.v_half = [
                np.asarray(p.velocity, dtype=float)
                + 0.5 * np.asarray(self.integrator.force_to_acceleration_func(p, self.particles), dtype=float)
                * self.integrator.dt
                for p in self.particles
            ]
        self.paused = False
        # reset pause button label
        for b in self._buttons:
            if b["callback"] is self._btn_pause_toggle:
                b["label"] = "Pause"

    def on_mouse_drag(self, x: float, y: float, dx: float, dy: float, buttons: int, modifiers: int):
        # Primary (left) drag -> pitch (x) by dy, yaw (y) by dx
        if buttons & arcade.MOUSE_BUTTON_LEFT and self._rotating_primary:
            self.rot_y += dx * self.rotation_sensitivity
            self.rot_x += -dy * self.rotation_sensitivity
            # normalize angles
            self.rot_x = (self.rot_x + math.pi) % (2 * math.pi) - math.pi
            self.rot_y = (self.rot_y + math.pi) % (2 * math.pi) - math.pi
        # Secondary (right) drag -> roll (z)
        if buttons & arcade.MOUSE_BUTTON_RIGHT and self._rotating_secondary:
            self.rot_z += dx * self.rotation_sensitivity
            self.rot_z = (self.rot_z + math.pi) % (2 * math.pi) - math.pi
        self._last_mouse_x = x
        self._last_mouse_y = y

    def on_mouse_scroll(self, x: float, y: float, scroll_x: float, scroll_y: float):
        # Use scroll to adjust camera distance (zoom equivalent in 3D)
        # scroll_y > 0 means scroll up -> zoom in
        factor = 0.9 ** (-scroll_y)
        self.camera_distance = float(np.clip(self.camera_distance * factor, self.min_camera_distance, self.max_camera_distance))

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
        if self.box is None or self.system is None:
            return
        # Delegate creation to System for consistency
        p = self.system.create_particle()
        self.particles.append(p)
        # Keep integrator internal lists consistent if needed
        if hasattr(self.integrator, "v_half"):
            # compute half-step velocity consistent with integrator's force function
            a = np.asarray(self.integrator.force_to_acceleration_func(p, self.integrator.particles), dtype=float)
            self.integrator.v_half.append(np.asarray(p.velocity) + 0.5 * a * self.integrator.dt)

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
