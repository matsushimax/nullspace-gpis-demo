"""Render README media from the real viewers using temporary example databases."""

import argparse
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from database import DatabaseBuildConfig, NullspaceDatabaseBuilder
from interactive_demo import InteractiveSelfMotionDemo
from obstacle_database.obstacle_manifold_database import (
    ObstacleManifoldDatabaseBuilder,
    load_nullspace_grid,
)
from planar_arm import PlanarArm3DOF
from robot.planar3r import Planar3RRobot
from scripts.view_database_interpolation import DatabaseInterpolationViewer
from scripts.view_obstacle_manifold_database import ObstacleManifoldViewer


def render_motion(output):
    demo = InteractiveSelfMotionDemo(PlanarArm3DOF(), (1.7, 0.65), max_samples=120)
    demo.fig.set_size_inches(11, 6)
    demo.fig.subplots_adjust(right=0.91, wspace=0.28)
    frames = []
    try:
        for phase in np.linspace(0, 2 * np.pi, 32, endpoint=False):
            demo.set_target(1.5 + 0.2 * np.cos(phase), 0.65 + 0.18 * np.sin(phase))
            demo.recompute_samples()
            if len(demo.samples) == 0:
                raise RuntimeError(demo.status)
            demo.redraw()
            demo.ax_manifold.view_init(elev=22, azim=40)
            demo.fig.canvas.draw()
            frames.append(Image.fromarray(np.asarray(demo.fig.canvas.buffer_rgba()).copy()).convert("RGB"))
        frames[0].save(
            output / "self_motion.gif", save_all=True, append_images=frames[1:],
            duration=125, loop=0, optimize=True,
        )
        frames[0].save(output / "self_motion.png")
    finally:
        plt.close(demo.fig)


def render_databases(output):
    with tempfile.TemporaryDirectory() as directory:
        null_path = Path(directory) / "nullspace.h5"
        obstacle_path = Path(directory) / "obstacles.h5"
        NullspaceDatabaseBuilder(PlanarArm3DOF()).build(
            null_path, DatabaseBuildConfig(grid_shape=(9, 9), max_samples=120, seed=42),
        )
        viewer = DatabaseInterpolationViewer(null_path, (1.48, 0.78), show_gt=True, k=1)
        try:
            if viewer.failure_reason:
                raise RuntimeError(viewer.failure_reason)
            viewer.fig.subplots_adjust(right=0.92, wspace=0.28)
            viewer.ax_manifold.view_init(elev=22, azim=40)
            for text in viewer.ax_workspace.texts:
                text.set_text(text.get_text().replace("  ", "\n", 2))
            viewer.fig.savefig(output / "tps_projection.png", dpi=120, bbox_inches="tight")
        finally:
            viewer.on_close(None)
            plt.close(viewer.fig)

        ObstacleManifoldDatabaseBuilder(Planar3RRobot()).build(
            obstacle_path, load_nullspace_grid(null_path), contact_epsilon=0.02,
            num_configurations=8000, max_boundary_pairs=35, seed=42,
        )
        viewer = ObstacleManifoldViewer(obstacle_path, (1.35, 0.675), num_arm_samples=5)
        try:
            if viewer.manifold is None or len(viewer.manifold.boundary_q) == 0:
                raise RuntimeError("No obstacle boundary samples available for the illustration")
            viewer.fig.subplots_adjust(right=0.92, wspace=0.28)
            for text in viewer.ax_manifold.texts:
                text.set_position((0.02, -0.08))
            for text in viewer.ax_workspace.texts:
                text.set_text(text.get_text().replace("  ", "\n", 2))
            viewer.fig.savefig(output / "obstacle_boundary.png", dpi=120, bbox_inches="tight")
        finally:
            viewer.on_close(None)
            plt.close(viewer.fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "images")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    render_motion(args.output)
    render_databases(args.output)
    for path in sorted(args.output.glob("*")):
        if path.is_file():
            print(f"{path.name}: {path.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
