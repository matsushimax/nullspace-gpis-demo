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
from deformation import interpolate_manifold_rbf, interpolate_manifold_tps
from planar_arm import PlanarArm3DOF
from sampler import jacobian_guided_sampling
from viewer_utils import source_colors


class DatabaseInterpolationViewer:
    def __init__(
        self,
        database_path,
        initial_target,
        k=1,
        radius_cells=2,
        anchors=16,
        method="tps-project",
        tps_regularization=1e-6,
        max_gt_samples=120,
        show_gt=False,
        update_interval=0.08,
    ):
        self.db = NullspaceDatabase.open(database_path)
        self.arm = PlanarArm3DOF(link_lengths=tuple(self.db.h5["meta"].attrs["link_lengths"]))
        self.target = np.asarray(initial_target, dtype=float)
        self.k = max(1, int(k))
        self.radius_cells = int(radius_cells)
        self.anchors = int(anchors)
        self.method = method
        self.tps_regularization = float(tps_regularization)
        self.max_gt_samples = int(max_gt_samples)
        self.show_gt = bool(show_gt)
        self.update_interval = float(update_interval)
        self.dragging = False
        self.last_update_time = 0.0

        self.results = []
        self.nearest = None
        self.extra_manifolds = []
        self.source_manifold = None
        self.deformation = None
        self.gt_q = None
        self.failure_reason = None
        self.estimate_time_ms = None
        self.gt_time_ms = None

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

        self.query_and_interpolate()
        self.redraw()

    def query_and_interpolate(self):
        self.failure_reason = None
        self.deformation = None
        self.gt_q = None
        self.estimate_time_ms = None
        self.gt_time_ms = None
        self.results = self.db.query_k_nearest_targets(
            self.target,
            k=self.k,
            search_radius_cells=self.radius_cells,
        )
        self.nearest = self.results[0] if self.results else None
        self.extra_manifolds = []
        if self.nearest is None:
            self.source_manifold = None
            self.failure_reason = "no nearby database target"
            return

        self.source_manifold = self.db.load_full_manifold(self.nearest.target_id)
        self.extra_manifolds = [self.db.load_full_manifold(result.target_id) for result in self.results[1:]]
        try:
            estimate_start = time.perf_counter()
            self.deformation = self.interpolate()
            self.estimate_time_ms = (time.perf_counter() - estimate_start) * 1000.0
        except Exception as exc:
            self.failure_reason = str(exc)

        if self.show_gt:
            try:
                gt_start = time.perf_counter()
                self.gt_q = jacobian_guided_sampling(
                    self.arm,
                    self.target,
                    beta=0.20,
                    max_samples=self.max_gt_samples,
                    seed=123,
                )
                self.gt_time_ms = (time.perf_counter() - gt_start) * 1000.0
            except Exception as exc:
                self.failure_reason = f"{self.failure_reason or ''} GT failed: {exc}".strip()

    def interpolate(self):
        if self.method == "rbf":
            return interpolate_manifold_rbf(
                self.arm,
                self.source_manifold,
                self.target,
                num_anchors=self.anchors,
                post_project=False,
            )
        if self.method == "rbf-project":
            return interpolate_manifold_rbf(
                self.arm,
                self.source_manifold,
                self.target,
                num_anchors=self.anchors,
                post_project=True,
            )
        if self.method == "tps":
            return interpolate_manifold_tps(
                self.arm,
                self.source_manifold,
                self.target,
                num_anchors=self.anchors,
                regularization=self.tps_regularization,
                post_project=False,
            )
        if self.method == "tps-project":
            return interpolate_manifold_tps(
                self.arm,
                self.source_manifold,
                self.target,
                num_anchors=self.anchors,
                regularization=self.tps_regularization,
                post_project=True,
            )
        raise ValueError(f"unknown interpolation method: {self.method}")

    def redraw(self):
        self.draw_workspace()
        self.draw_manifold()
        self.fig.canvas.draw_idle()

    def draw_workspace(self):
        ax = self.ax_workspace
        ax.clear()
        ax.set_title("Workspace query and interpolation source")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.30)
        ax.set_xlim(float(self.db.workspace_min[0]), float(self.db.workspace_max[0]))
        ax.set_ylim(float(self.db.workspace_min[1]), float(self.db.workspace_max[1]))

        if len(self.all_positions):
            ax.scatter(self.all_positions[:, 0], self.all_positions[:, 1], s=12, color="0.84", label="all targets")
        if len(self.valid_positions):
            ax.scatter(self.valid_positions[:, 0], self.valid_positions[:, 1], s=18, color="0.45", label="valid targets")

        if self.nearest:
            colors = source_colors(len(self.results))
            source = self.nearest.target
            ax.scatter([source[0]], [source[1]], s=145, color=colors[0], label="nearest source", zorder=5)
            ax.plot(
                [self.target[0], source[0]],
                [self.target[1], source[1]],
                color=colors[0],
                alpha=0.45,
                linewidth=1.2,
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

        ax.scatter([self.target[0]], [self.target[1]], marker="x", s=140, color="tab:red", label="query", zorder=6)

        status = f"query=({self.target[0]:.2f}, {self.target[1]:.2f})"
        if self.nearest:
            status += f"  source id={self.nearest.target_id}  dist={self.nearest.distance:.3f}"
        if self.deformation is not None:
            post = " + project" if self.deformation.post_projected else ""
            status += (
                f"  anchors={len(self.deformation.anchor_result.source_indices)}"
                f"  method={self.deformation.method}{post}"
                f"  RMSE={np.sqrt(np.mean(self.deformation.task_residuals ** 2)):.2e}"
            )
        if self.failure_reason:
            status += f"  {self.failure_reason}"
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
        ax.set_title(f"{self.method} null-space manifold")

        if self.source_manifold is None:
            ax.text2D(0.05, 0.95, "No source manifold", transform=ax.transAxes)
            return

        q_source = self.source_manifold.q
        colors = source_colors(1 + len(self.extra_manifolds))
        ax.plot(
            q_source[:, 0],
            q_source[:, 1],
            q_source[:, 2],
            marker="o",
            markersize=2.5,
            linewidth=1.0,
            color=colors[0],
            alpha=0.60,
            label="source",
        )
        for index, manifold in enumerate(self.extra_manifolds, start=1):
            q_extra = manifold.q
            if len(q_extra) == 0:
                continue
            ax.plot(
                q_extra[:, 0],
                q_extra[:, 1],
                q_extra[:, 2],
                marker="o",
                markersize=2.1,
                linewidth=0.9,
                color=colors[index],
                alpha=0.45,
                label=f"neighbor {index + 1}",
            )

        if self.deformation is not None:
            q_est = self.deformation.estimated_q
            anchors = self.deformation.anchor_result
            ax.plot(
                q_est[:, 0],
                q_est[:, 1],
                q_est[:, 2],
                marker="o",
                markersize=3,
                linewidth=1.4,
                color="tab:purple",
                label=f"estimated {self.method}",
            )
            ax.scatter(
                anchors.source_anchors[:, 0],
                anchors.source_anchors[:, 1],
                anchors.source_anchors[:, 2],
                s=35,
                color="tab:blue",
                label="source anchors",
            )
            ax.scatter(
                anchors.projected_anchors[:, 0],
                anchors.projected_anchors[:, 1],
                anchors.projected_anchors[:, 2],
                s=45,
                color="tab:red",
                label="projected anchors",
            )

        if self.gt_q is not None and len(self.gt_q):
            ax.plot(
                self.gt_q[:, 0],
                self.gt_q[:, 1],
                self.gt_q[:, 2],
                marker=".",
                markersize=3,
                linewidth=1.0,
                color="tab:green",
                alpha=0.70,
                label="direct GT",
            )

        ax.text2D(
            0.02,
            -0.14,
            self.metric_text(),
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            clip_on=False,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.78, "edgecolor": "0.8"},
        )
        ax.legend(loc="upper left", fontsize=8)

    def metric_text(self):
        lines = []
        if self.estimate_time_ms is not None:
            method = self.deformation.method if self.deformation is not None else self.method
            post = " + project" if self.deformation is not None and self.deformation.post_projected else ""
            lines.append(f"estimated {method}{post}: {self.estimate_time_ms:.1f} ms")
        else:
            lines.append("estimated: n/a")

        if self.gt_time_ms is not None:
            lines.append(f"direct GT: {self.gt_time_ms:.1f} ms")
        elif self.show_gt:
            lines.append("direct GT: n/a")
        else:
            lines.append("direct GT: off")

        if self.deformation is not None and len(self.deformation.task_residuals):
            errors = self.deformation.task_residuals
            lines.append(
                "task error | "
                f"mean={np.mean(errors):.2e}  "
                f"max={np.max(errors):.2e}  "
                f"var={np.var(errors):.2e}"
            )
        else:
            lines.append("task error: n/a")
        return "\n".join(lines)

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
        self.query_and_interpolate()
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
        self.query_and_interpolate()
        self.redraw()

    def on_close(self, _):
        self.db.close()

    def show(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Interactively visualize RBF manifold interpolation from the database.")
    parser.add_argument("--database", default="data/nullspace_database.h5", help="HDF5 database path")
    parser.add_argument("--target", nargs=2, type=float, default=(1.3, 0.7), metavar=("X", "Y"))
    parser.add_argument("-k", "--k", type=int, default=1, help="draw the k nearest source manifolds")
    parser.add_argument("--radius-cells", type=int, default=2, help="neighbor cell search radius")
    parser.add_argument("--anchors", type=int, default=16, help="number of source anchors to project")
    parser.add_argument(
        "--method",
        choices=("rbf", "rbf-project", "tps", "tps-project"),
        default="tps-project",
        help="interpolation method; *-project runs a final Gauss-Newton projection for every estimated sample",
    )
    parser.add_argument("--tps-regularization", type=float, default=1e-6, help="TPS regularization")
    parser.add_argument("--show-gt", action="store_true", help="also draw direct self-motion samples for the query target")
    parser.add_argument("--max-gt-samples", type=int, default=120, help="maximum direct GT samples")
    args = parser.parse_args()

    viewer = DatabaseInterpolationViewer(
        database_path=args.database,
        initial_target=args.target,
        k=args.k,
        radius_cells=args.radius_cells,
        anchors=args.anchors,
        method=args.method,
        tps_regularization=args.tps_regularization,
        show_gt=args.show_gt,
        max_gt_samples=args.max_gt_samples,
    )
    viewer.show()


if __name__ == "__main__":
    main()
