import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.widgets import Slider

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from obstacle_database.obstacle_manifold_database import ObstacleManifoldDatabase
from robot.planar3r import Planar3RRobot


class ObstacleManifoldViewer:
    def __init__(self, database_path, initial_center=(0.8, 0.4), num_arm_samples=6, update_interval=0.05):
        self.db = ObstacleManifoldDatabase.open(database_path)
        self.robot = self.load_robot()
        self.center = np.asarray(initial_center, dtype=float)
        self.num_arm_samples = max(0, int(num_arm_samples))
        self.update_interval = float(update_interval)
        self.dragging = False
        self.last_update_time = 0.0
        self.query = None
        self.manifold = None

        centers = self.db.h5["obstacles/centers"][:]
        valid = self.db.h5["obstacles/valid"][:].astype(bool)
        self.all_centers = centers
        self.valid_centers = centers[valid]

        self.fig = plt.figure(figsize=(13, 7))
        self.ax_workspace = self.fig.add_subplot(121)
        self.ax_manifold = self.fig.add_subplot(122, projection="3d")
        self.fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, wspace=0.18)

        x_ax = self.fig.add_axes([0.12, 0.08, 0.30, 0.025])
        y_ax = self.fig.add_axes([0.12, 0.04, 0.30, 0.025])
        self.x_slider = Slider(
            x_ax,
            "obstacle x",
            float(self.db.workspace_min[0]),
            float(self.db.workspace_max[0]),
            valinit=self.center[0],
        )
        self.y_slider = Slider(
            y_ax,
            "obstacle y",
            float(self.db.workspace_min[1]),
            float(self.db.workspace_max[1]),
            valinit=self.center[1],
        )
        self.x_slider.on_changed(self.on_slider_changed)
        self.y_slider.on_changed(self.on_slider_changed)
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("close_event", self.on_close)

        self.query_database()
        self.redraw()

    def load_robot(self):
        attrs = self.db.h5["meta"].attrs
        link_lengths = json.loads(attrs["link_lengths"]) if "link_lengths" in attrs else (1.0, 1.0, 0.7)
        joint_limits = json.loads(attrs["joint_limits"]) if "joint_limits" in attrs else None
        return Planar3RRobot(
            link_lengths=link_lengths,
            joint_limits=joint_limits,
            link_radius=0.06,
            circle_radius=0.06,
            circle_spacing=0.08,
        )

    def query_database(self):
        self.query = self.db.query_nearest_obstacle(self.center)
        self.manifold = self.db.load_manifold(self.query[0]) if self.query else None

    def picked_boundary_indices(self):
        if self.manifold is None or len(self.manifold.boundary_q) == 0 or self.num_arm_samples <= 0:
            return np.empty((0,), dtype=np.int32)
        count = min(self.num_arm_samples, len(self.manifold.boundary_q))
        return np.linspace(0, len(self.manifold.boundary_q) - 1, count).round().astype(np.int32)

    def picked_colors(self, count):
        if count <= 0:
            return []
        return plt.cm.tab10(np.arange(count) % 10)

    def redraw(self):
        self.draw_workspace()
        self.draw_manifold()
        self.fig.canvas.draw_idle()

    def draw_workspace(self):
        ax = self.ax_workspace
        ax.clear()
        ax.set_title("Obstacle center grid")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.30)
        ax.set_xlim(float(self.db.workspace_min[0]), float(self.db.workspace_max[0]))
        ax.set_ylim(float(self.db.workspace_min[1]), float(self.db.workspace_max[1]))

        if len(self.all_centers):
            ax.scatter(self.all_centers[:, 0], self.all_centers[:, 1], s=12, color="0.84", label="all centers")
        if len(self.valid_centers):
            ax.scatter(self.valid_centers[:, 0], self.valid_centers[:, 1], s=18, color="0.45", label="valid centers")

        ax.scatter([self.center[0]], [self.center[1]], marker="x", s=140, color="tab:red", label="query obstacle point")
        if self.db.contact_epsilon > 0:
            ax.add_patch(
                Circle(
                    self.center,
                    self.db.contact_epsilon,
                    color="tab:red",
                    fill=False,
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.65,
                )
            )
        if self.query:
            obstacle_id, source, distance = self.query
            ax.scatter([source[0]], [source[1]], s=145, color="tab:orange", label="nearest obstacle point")
            if self.db.contact_epsilon > 0:
                ax.add_patch(
                    Circle(
                        source,
                        self.db.contact_epsilon,
                        color="tab:orange",
                        fill=False,
                        linewidth=1.0,
                        linestyle="--",
                        alpha=0.70,
                    )
                )
            ax.plot([self.center[0], source[0]], [self.center[1], source[1]], color="tab:orange", alpha=0.45)
            self.draw_picked_arms(ax)
            status = (
                f"query=({self.center[0]:.2f}, {self.center[1]:.2f})  "
                f"source id={obstacle_id}  dist={distance:.3f}  eps={self.db.contact_epsilon:.3f}  "
                f"samples={len(self.manifold.boundary_q)}"
            )
        else:
            status = f"query=({self.center[0]:.2f}, {self.center[1]:.2f})  no obstacle manifold"
        ax.text(
            0.02,
            0.02,
            status,
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.78, "edgecolor": "0.8"},
        )
        ax.legend(loc="upper right", fontsize=8)

    def draw_picked_arms(self, ax):
        indices = self.picked_boundary_indices()
        colors = self.picked_colors(len(indices))
        for draw_index, sample_index in enumerate(indices):
            q = self.manifold.boundary_q[sample_index]
            pts = self.robot.forward_kinematics(q)
            color = colors[draw_index]
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                color=color,
                alpha=0.85,
                label="picked arm" if draw_index == 0 else None,
                zorder=6,
            )
            closest, distance, _ = self.robot.closest_link_point(q, self.manifold.center)
            ax.scatter([closest[0]], [closest[1]], s=28, color=color, edgecolors="black", linewidths=0.4, zorder=7)
            ax.plot(
                [self.manifold.center[0], closest[0]],
                [self.manifold.center[1], closest[1]],
                color=color,
                linestyle=":",
                linewidth=1.0,
                alpha=0.9,
                zorder=7,
            )

    def draw_manifold(self):
        ax = self.ax_manifold
        ax.clear()
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_zlim(-np.pi, np.pi)
        ax.set_xlabel("q1 [rad]")
        ax.set_ylabel("q2 [rad]")
        ax.set_zlabel("q3 [rad]")
        ax.set_title("Obstacle collision-boundary manifold")

        if self.manifold is None or len(self.manifold.boundary_q) == 0:
            ax.text2D(0.05, 0.95, "No boundary samples", transform=ax.transAxes)
            return
        q = self.manifold.boundary_q
        ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=12, color="tab:orange", label="collision boundary")
        indices = self.picked_boundary_indices()
        colors = self.picked_colors(len(indices))
        for draw_index, sample_index in enumerate(indices):
            q_pick = q[sample_index]
            ax.scatter(
                [q_pick[0]],
                [q_pick[1]],
                [q_pick[2]],
                s=70,
                color=colors[draw_index],
                edgecolors="black",
                linewidths=0.5,
                label="picked boundary" if draw_index == 0 else None,
                zorder=5,
            )
        if len(self.manifold.source_free_q):
            f = self.manifold.source_free_q
            ax.scatter(f[:, 0], f[:, 1], f[:, 2], s=8, color="tab:green", alpha=0.35, label="free side")
        if len(self.manifold.source_collision_q):
            c = self.manifold.source_collision_q
            ax.scatter(c[:, 0], c[:, 1], c[:, 2], s=8, color="tab:red", alpha=0.35, label="collision side")
        ax.text2D(
            0.05,
            0.95,
            f"obstacle id={self.manifold.obstacle_id}  point=({self.manifold.center[0]:.2f}, {self.manifold.center[1]:.2f})  eps={self.db.contact_epsilon:.3f}",
            transform=ax.transAxes,
        )
        ax.legend(loc="upper left", fontsize=8)

    def set_center(self, x, y, sync_sliders=False):
        self.center = np.array(
            [
                np.clip(x, self.db.workspace_min[0], self.db.workspace_max[0]),
                np.clip(y, self.db.workspace_min[1], self.db.workspace_max[1]),
            ],
            dtype=float,
        )
        if sync_sliders:
            self.sync_sliders()

    def sync_sliders(self):
        x_event_state = self.x_slider.eventson
        y_event_state = self.y_slider.eventson
        self.x_slider.eventson = False
        self.y_slider.eventson = False
        self.x_slider.set_val(self.center[0])
        self.y_slider.set_val(self.center[1])
        self.x_slider.eventson = x_event_state
        self.y_slider.eventson = y_event_state

    def update_query(self, x, y, force=False):
        now = time.monotonic()
        if not force and now - self.last_update_time < self.update_interval:
            return
        self.last_update_time = now
        self.set_center(x, y)
        self.query_database()
        self.redraw()

    def on_press(self, event):
        if event.inaxes != self.ax_workspace or event.xdata is None or event.ydata is None:
            return
        self.dragging = True
        self.update_query(event.xdata, event.ydata, force=True)

    def on_motion(self, event):
        if not self.dragging or event.inaxes != self.ax_workspace or event.xdata is None or event.ydata is None:
            return
        self.update_query(event.xdata, event.ydata)

    def on_release(self, event):
        if not self.dragging:
            return
        self.dragging = False
        if event.inaxes == self.ax_workspace and event.xdata is not None and event.ydata is not None:
            self.update_query(event.xdata, event.ydata, force=True)
        self.sync_sliders()

    def on_slider_changed(self, _):
        if self.dragging:
            return
        self.set_center(self.x_slider.val, self.y_slider.val)
        self.query_database()
        self.redraw()

    def on_close(self, _):
        self.db.close()

    def show(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Interactively view obstacle collision-boundary manifolds.")
    parser.add_argument("--database", default="data/obstacle_manifold_database.h5")
    parser.add_argument("--center", nargs=2, type=float, default=(0.8, 0.4), metavar=("X", "Y"))
    parser.add_argument("--arm-samples", type=int, default=6, help="number of boundary arm postures to draw")
    args = parser.parse_args()

    viewer = ObstacleManifoldViewer(args.database, initial_center=args.center, num_arm_samples=args.arm_samples)
    viewer.show()


if __name__ == "__main__":
    main()
