import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database import NullspaceDatabase
from viewer_utils import source_colors


class DatabaseQueryViewer:
    def __init__(self, database_path, initial_target, k=1, radius_cells=2, update_interval=0.05):
        self.db = NullspaceDatabase.open(database_path)
        self.target = np.asarray(initial_target, dtype=float)
        self.k = max(1, int(k))
        self.radius_cells = int(radius_cells)
        self.update_interval = float(update_interval)
        self.dragging = False
        self.last_update_time = 0.0
        self.results = []
        self.nearest = None
        self.manifold = None
        self.extra_manifolds = []

        positions = self.db.h5["targets/positions"][:]
        valid = self.db.h5["targets/valid"][:]
        self.valid_positions = positions[valid]
        self.all_positions = positions

        self.fig = plt.figure(figsize=(13, 7))
        self.ax_workspace = self.fig.add_subplot(121)
        self.ax_manifold = self.fig.add_subplot(122, projection="3d")
        self.fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, wspace=0.18)

        x_ax = self.fig.add_axes([0.12, 0.08, 0.30, 0.025])
        y_ax = self.fig.add_axes([0.12, 0.04, 0.30, 0.025])
        self.x_slider = Slider(
            x_ax,
            "target x",
            float(self.db.workspace_min[0]),
            float(self.db.workspace_max[0]),
            valinit=self.target[0],
        )
        self.y_slider = Slider(
            y_ax,
            "target y",
            float(self.db.workspace_min[1]),
            float(self.db.workspace_max[1]),
            valinit=self.target[1],
        )
        self.x_slider.on_changed(self.on_slider_changed)
        self.y_slider.on_changed(self.on_slider_changed)
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("close_event", self.on_close)

        self.query_database()
        self.redraw()

    def query_database(self):
        self.results = self.db.query_k_nearest_targets(
            self.target,
            k=self.k,
            search_radius_cells=self.radius_cells,
        )
        self.nearest = self.results[0] if self.results else None
        self.manifold = self.db.load_full_manifold(self.nearest.target_id) if self.nearest else None
        self.extra_manifolds = [self.db.load_full_manifold(result.target_id) for result in self.results[1:]]

    def redraw(self):
        self.draw_workspace()
        self.draw_manifold()
        self.fig.canvas.draw_idle()

    def draw_workspace(self):
        ax = self.ax_workspace
        ax.clear()
        ax.set_title("Workspace hash query")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.30)
        ax.set_xlim(float(self.db.workspace_min[0]), float(self.db.workspace_max[0]))
        ax.set_ylim(float(self.db.workspace_min[1]), float(self.db.workspace_max[1]))

        if len(self.all_positions):
            ax.scatter(
                self.all_positions[:, 0],
                self.all_positions[:, 1],
                s=12,
                color="0.82",
                label="all grid targets",
            )
        if len(self.valid_positions):
            ax.scatter(
                self.valid_positions[:, 0],
                self.valid_positions[:, 1],
                s=18,
                color="0.45",
                label="valid DB targets",
            )

        if self.nearest:
            colors = source_colors(len(self.results))
            source = self.nearest.target
            ax.scatter([source[0]], [source[1]], s=140, color=colors[0], label="nearest source", zorder=5)
            ax.plot(
                [self.target[0], source[0]],
                [self.target[1], source[1]],
                color=colors[0],
                alpha=0.45,
                linewidth=1.2,
            )
            status = (
                f"query=({self.target[0]:.2f}, {self.target[1]:.2f})  "
                f"source id={self.nearest.target_id}  "
                f"dist={self.nearest.distance:.3f}  samples={self.nearest.num_samples}"
            )
            if len(self.results) > 1:
                for index, result in enumerate(self.results[1:], start=1):
                    ax.scatter(
                        [result.target[0]],
                        [result.target[1]],
                        s=75,
                        color=colors[index],
                        label=f"source {index + 1}",
                        zorder=4,
                    )
        else:
            status = f"query=({self.target[0]:.2f}, {self.target[1]:.2f})  no target found"

        ax.scatter([self.target[0]], [self.target[1]], marker="x", s=140, color="tab:red", label="query", zorder=6)
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

    def draw_manifold(self):
        ax = self.ax_manifold
        ax.clear()
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_zlim(-np.pi, np.pi)
        ax.set_xlabel("q1 [rad]")
        ax.set_ylabel("q2 [rad]")
        ax.set_zlabel("q3 [rad]")
        ax.set_title("Referenced null-space manifold")

        if self.manifold is None:
            ax.text2D(0.05, 0.95, "No manifold loaded", transform=ax.transAxes)
            return

        q = self.manifold.q
        colors = source_colors(1 + len(self.extra_manifolds))
        ax.plot(
            q[:, 0],
            q[:, 1],
            q[:, 2],
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=colors[0],
            label=f"source id={self.manifold.target_id}",
        )
        ax.scatter([q[0, 0]], [q[0, 1]], [q[0, 2]], s=65, color="tab:red", label="first sample")
        for index, manifold in enumerate(self.extra_manifolds, start=1):
            q_extra = manifold.q
            if len(q_extra) == 0:
                continue
            ax.plot(
                q_extra[:, 0],
                q_extra[:, 1],
                q_extra[:, 2],
                marker="o",
                markersize=2.2,
                linewidth=0.9,
                color=colors[index],
                alpha=0.45,
                label=f"neighbor {index + 1} id={manifold.target_id}",
            )
        ax.text2D(
            0.05,
            0.95,
            f"target_id={self.manifold.target_id}  target=({self.manifold.target[0]:.2f}, {self.manifold.target[1]:.2f})",
            transform=ax.transAxes,
        )
        ax.legend(loc="upper left", fontsize=8)

    def set_target(self, x, y, sync_sliders=False):
        self.target = np.array(
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
        self.x_slider.set_val(self.target[0])
        self.y_slider.set_val(self.target[1])
        self.x_slider.eventson = x_event_state
        self.y_slider.eventson = y_event_state

    def update_query(self, x, y, force=False):
        now = time.monotonic()
        if not force and now - self.last_update_time < self.update_interval:
            return
        self.last_update_time = now
        self.set_target(x, y)
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
        self.set_target(self.x_slider.val, self.y_slider.val)
        self.query_database()
        self.redraw()

    def on_close(self, _):
        self.db.close()

    def show(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Interactively visualize workspace hash database queries.")
    parser.add_argument("--database", default="data/nullspace_database.h5", help="HDF5 database path")
    parser.add_argument("--target", nargs=2, type=float, default=(1.3, 0.7), metavar=("X", "Y"))
    parser.add_argument("-k", "--k", type=int, default=1, help="draw the k nearest source manifolds")
    parser.add_argument("--radius-cells", type=int, default=2, help="neighbor cell search radius")
    args = parser.parse_args()

    viewer = DatabaseQueryViewer(
        database_path=args.database,
        initial_target=args.target,
        k=args.k,
        radius_cells=args.radius_cells,
    )
    viewer.show()


if __name__ == "__main__":
    main()
