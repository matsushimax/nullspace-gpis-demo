import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database import NullspaceDatabase
from deformation import interpolate_manifold_rbf, interpolate_manifold_tps
from obstacle_database.obstacle_manifold_database import ObstacleManifoldDatabase
from planar_arm import PlanarArm3DOF
from robot.planar3r import Planar3RRobot
from sampler import jacobian_guided_sampling


class CombinedManifoldViewer:
    def __init__(
        self,
        nullspace_database,
        obstacle_database,
        initial_target=(1.3, 0.7),
        initial_obstacle=(0.8, 0.4),
        method="tps-project",
        anchors=16,
        radius_cells=2,
        show_gt=False,
        max_gt_samples=120,
        update_interval=0.05,
    ):
        self.null_db = NullspaceDatabase.open(nullspace_database)
        self.obs_db = ObstacleManifoldDatabase.open(obstacle_database)
        self.arm = PlanarArm3DOF(link_lengths=tuple(self.null_db.h5["meta"].attrs["link_lengths"]))
        self.robot = Planar3RRobot(link_lengths=tuple(self.null_db.h5["meta"].attrs["link_lengths"]))
        self.target = np.asarray(initial_target, dtype=float)
        self.obstacle = np.asarray(initial_obstacle, dtype=float)
        self.method = method
        self.anchors = int(anchors)
        self.radius_cells = int(radius_cells)
        self.show_gt = bool(show_gt)
        self.max_gt_samples = int(max_gt_samples)
        self.update_interval = float(update_interval)
        self.drag_mode = None
        self.last_update_time = 0.0

        self.null_result = None
        self.source_manifold = None
        self.deformation = None
        self.gt_q = None
        self.obstacle_query = None
        self.obstacle_manifold = None
        self.estimate_time_ms = None
        self.gt_time_ms = None
        self.failure_reason = None

        self.null_targets = self.null_db.h5["targets/positions"][:]
        self.null_valid_targets = self.null_targets[self.null_db.h5["targets/valid"][:].astype(bool)]
        self.obs_centers = self.obs_db.h5["obstacles/centers"][:]
        self.obs_valid_centers = self.obs_centers[self.obs_db.h5["obstacles/valid"][:].astype(bool)]

        self.fig = plt.figure(figsize=(13, 7))
        self.ax_workspace = self.fig.add_subplot(121)
        self.ax_manifold = self.fig.add_subplot(122, projection="3d")
        self.fig.subplots_adjust(left=0.07, right=0.98, bottom=0.10, wspace=0.18)

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("close_event", self.on_close)

        self.recompute()
        self.redraw()

    def recompute(self):
        self.failure_reason = None
        self.deformation = None
        self.gt_q = None
        self.estimate_time_ms = None
        self.gt_time_ms = None

        self.null_result = self.null_db.query_nearest_target(self.target, search_radius_cells=self.radius_cells)
        if self.null_result is None:
            self.source_manifold = None
            self.failure_reason = "no null-space source"
        else:
            self.source_manifold = self.null_db.load_full_manifold(self.null_result.target_id)
            try:
                start = time.perf_counter()
                self.deformation = self.interpolate_nullspace()
                self.estimate_time_ms = (time.perf_counter() - start) * 1000.0
            except Exception as exc:
                self.failure_reason = str(exc)

        self.obstacle_query = self.obs_db.query_nearest_obstacle(self.obstacle)
        self.obstacle_manifold = self.obs_db.load_manifold(self.obstacle_query[0]) if self.obstacle_query else None

        if self.show_gt:
            try:
                start = time.perf_counter()
                self.gt_q = jacobian_guided_sampling(
                    self.arm,
                    self.target,
                    beta=0.20,
                    max_samples=self.max_gt_samples,
                    seed=123,
                )
                self.gt_time_ms = (time.perf_counter() - start) * 1000.0
            except Exception as exc:
                self.failure_reason = f"{self.failure_reason or ''} GT failed: {exc}".strip()

    def interpolate_nullspace(self):
        if self.method == "rbf":
            return interpolate_manifold_rbf(self.arm, self.source_manifold, self.target, num_anchors=self.anchors)
        if self.method == "rbf-project":
            return interpolate_manifold_rbf(
                self.arm, self.source_manifold, self.target, num_anchors=self.anchors, post_project=True
            )
        if self.method == "tps":
            return interpolate_manifold_tps(self.arm, self.source_manifold, self.target, num_anchors=self.anchors)
        if self.method == "tps-project":
            return interpolate_manifold_tps(
                self.arm, self.source_manifold, self.target, num_anchors=self.anchors, post_project=True
            )
        raise ValueError(f"unknown method: {self.method}")

    def redraw(self):
        self.draw_workspace()
        self.draw_manifold()
        self.fig.canvas.draw_idle()

    def draw_workspace(self):
        ax = self.ax_workspace
        ax.clear()
        ax.set_title("Left click: end-effector target / Right click: obstacle")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.30)
        ax.set_xlim(float(self.null_db.workspace_min[0]), float(self.null_db.workspace_max[0]))
        ax.set_ylim(float(self.null_db.workspace_min[1]), float(self.null_db.workspace_max[1]))

        if len(self.null_targets):
            ax.scatter(self.null_targets[:, 0], self.null_targets[:, 1], s=9, color="0.88", label="null grid")
        if len(self.null_valid_targets):
            ax.scatter(
                self.null_valid_targets[:, 0],
                self.null_valid_targets[:, 1],
                s=13,
                color="0.58",
                alpha=0.55,
                label="valid targets",
            )
        if len(self.obs_valid_centers):
            ax.scatter(
                self.obs_valid_centers[:, 0],
                self.obs_valid_centers[:, 1],
                s=10,
                color="tab:cyan",
                alpha=0.22,
                label="obstacle manifolds",
            )

        ax.scatter([self.target[0]], [self.target[1]], marker="x", s=145, color="tab:red", label="target")
        if self.null_result:
            source = self.null_result.target
            ax.scatter([source[0]], [source[1]], s=115, color="tab:orange", label="null source")
            ax.plot([self.target[0], source[0]], [self.target[1], source[1]], color="tab:orange", alpha=0.45)

        ax.scatter([self.obstacle[0]], [self.obstacle[1]], marker="x", s=145, color="tab:purple", label="obstacle point")
        if self.obs_db.contact_epsilon > 0:
            ax.add_patch(
                Circle(
                    self.obstacle,
                    self.obs_db.contact_epsilon,
                    color="tab:purple",
                    fill=False,
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.65,
                )
            )
        if self.obstacle_query:
            _, source, _ = self.obstacle_query
            ax.scatter([source[0]], [source[1]], s=110, color="tab:blue", label="obstacle source")
            if self.obs_db.contact_epsilon > 0:
                ax.add_patch(
                    Circle(
                        source,
                        self.obs_db.contact_epsilon,
                        color="tab:blue",
                        fill=False,
                        linewidth=1.0,
                        linestyle="--",
                        alpha=0.65,
                    )
                )
            ax.plot([self.obstacle[0], source[0]], [self.obstacle[1], source[1]], color="tab:blue", alpha=0.40)

        ax.text(
            0.02,
            0.02,
            self.workspace_status_text(),
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
        ax.set_title("Null-space and obstacle manifolds")

        if self.source_manifold is not None and len(self.source_manifold.q):
            q_source = self.source_manifold.q
            ax.plot(
                q_source[:, 0],
                q_source[:, 1],
                q_source[:, 2],
                marker="o",
                markersize=2.2,
                linewidth=0.9,
                color="tab:orange",
                alpha=0.45,
                label="null source",
            )
        if self.deformation is not None and len(self.deformation.estimated_q):
            q_est = self.deformation.estimated_q
            ax.plot(
                q_est[:, 0],
                q_est[:, 1],
                q_est[:, 2],
                marker="o",
                markersize=3,
                linewidth=1.4,
                color="tab:purple",
                label=f"null estimated {self.method}",
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
                alpha=0.65,
                label="direct null GT",
            )
        if self.obstacle_manifold is not None and len(self.obstacle_manifold.boundary_q):
            q_obs = self.obstacle_manifold.boundary_q
            ax.scatter(
                q_obs[:, 0],
                q_obs[:, 1],
                q_obs[:, 2],
                s=16,
                color="tab:blue",
                alpha=0.75,
                label="obstacle boundary",
            )

        ax.text2D(
            0.02,
            -0.12,
            self.metric_text(),
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            clip_on=False,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.78, "edgecolor": "0.8"},
        )
        ax.legend(loc="upper left", fontsize=8)

    def workspace_status_text(self):
        parts = [f"target=({self.target[0]:.2f}, {self.target[1]:.2f})"]
        if self.null_result:
            parts.append(f"null source id={self.null_result.target_id} dist={self.null_result.distance:.3f}")
        parts.append(f"obstacle=({self.obstacle[0]:.2f}, {self.obstacle[1]:.2f})")
        if self.obstacle_query:
            parts.append(f"obs source id={self.obstacle_query[0]} dist={self.obstacle_query[2]:.3f}")
        if self.failure_reason:
            parts.append(self.failure_reason)
        return "  ".join(parts)

    def metric_text(self):
        lines = []
        if self.deformation is not None and len(self.deformation.task_residuals):
            err = self.deformation.task_residuals
            lines.append(
                f"null estimated: {self.estimate_time_ms:.1f} ms  "
                f"mean={np.mean(err):.2e} max={np.max(err):.2e}"
            )
        else:
            lines.append("null estimated: n/a")
        if self.gt_time_ms is not None:
            lines.append(f"direct null GT: {self.gt_time_ms:.1f} ms")
        elif self.show_gt:
            lines.append("direct null GT: n/a")
        if self.obstacle_manifold is not None:
            lines.append(f"obstacle boundary samples: {len(self.obstacle_manifold.boundary_q)}")
        else:
            lines.append("obstacle boundary samples: n/a")
        return "\n".join(lines)

    def set_target(self, x, y):
        self.target = np.array(
            [
                np.clip(x, self.null_db.workspace_min[0], self.null_db.workspace_max[0]),
                np.clip(y, self.null_db.workspace_min[1], self.null_db.workspace_max[1]),
            ],
            dtype=float,
        )

    def set_obstacle(self, x, y):
        self.obstacle = np.array(
            [
                np.clip(x, self.obs_db.workspace_min[0], self.obs_db.workspace_max[0]),
                np.clip(y, self.obs_db.workspace_min[1], self.obs_db.workspace_max[1]),
            ],
            dtype=float,
        )

    def update_query(self, x, y, mode, force=False):
        now = time.monotonic()
        if not force and now - self.last_update_time < self.update_interval:
            return
        self.last_update_time = now
        if mode == "target":
            self.set_target(x, y)
        elif mode == "obstacle":
            self.set_obstacle(x, y)
        self.recompute()
        self.redraw()

    def on_press(self, event):
        if event.inaxes != self.ax_workspace or event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            self.drag_mode = "target"
        elif event.button == 3:
            self.drag_mode = "obstacle"
        else:
            return
        self.update_query(event.xdata, event.ydata, self.drag_mode, force=True)

    def on_motion(self, event):
        if self.drag_mode is None or event.inaxes != self.ax_workspace or event.xdata is None or event.ydata is None:
            return
        self.update_query(event.xdata, event.ydata, self.drag_mode)

    def on_release(self, event):
        if self.drag_mode is None:
            return
        mode = self.drag_mode
        self.drag_mode = None
        if event.inaxes == self.ax_workspace and event.xdata is not None and event.ydata is not None:
            self.update_query(event.xdata, event.ydata, mode, force=True)

    def on_close(self, _):
        self.null_db.close()
        self.obs_db.close()

    def show(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Interactively view null-space and obstacle manifolds together.")
    parser.add_argument("--nullspace-database", default="data/nullspace_database.h5")
    parser.add_argument("--obstacle-database", default="data/obstacle_manifold_database.h5")
    parser.add_argument("--target", nargs=2, type=float, default=(1.3, 0.7), metavar=("X", "Y"))
    parser.add_argument("--obstacle", nargs=2, type=float, default=(0.8, 0.4), metavar=("X", "Y"))
    parser.add_argument("--method", choices=("rbf", "rbf-project", "tps", "tps-project"), default="tps-project")
    parser.add_argument("--anchors", type=int, default=16)
    parser.add_argument("--radius-cells", type=int, default=2)
    parser.add_argument("--show-gt", action="store_true")
    args = parser.parse_args()

    viewer = CombinedManifoldViewer(
        nullspace_database=args.nullspace_database,
        obstacle_database=args.obstacle_database,
        initial_target=args.target,
        initial_obstacle=args.obstacle,
        method=args.method,
        anchors=args.anchors,
        radius_cells=args.radius_cells,
        show_gt=args.show_gt,
    )
    viewer.show()


if __name__ == "__main__":
    main()
