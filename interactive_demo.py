import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from planar_arm import PlanarArm3DOF
from sampler import jacobian_guided_sampling


class InteractiveSelfMotionDemo:
    def __init__(
        self,
        arm,
        initial_target,
        beta=0.20,
        max_samples=180,
        seed=42,
        update_interval=0.08,
    ):
        self.arm = arm
        self.target = np.asarray(initial_target, dtype=float)
        self.beta = float(beta)
        self.max_samples = int(max_samples)
        self.seed = int(seed)
        self.update_interval = float(update_interval)
        self.dragging = False
        self.last_update_time = 0.0
        self.samples = np.empty((0, 3))
        self.status = ""

        self.reach = float(np.sum(self.arm.link_lengths))
        self.fig = plt.figure(figsize=(13, 7))
        self.ax_arm = self.fig.add_subplot(121)
        self.ax_manifold = self.fig.add_subplot(122, projection="3d")
        self.fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, wspace=0.18)

        slider_x_ax = self.fig.add_axes([0.12, 0.08, 0.30, 0.025])
        slider_y_ax = self.fig.add_axes([0.12, 0.04, 0.30, 0.025])
        reset_ax = self.fig.add_axes([0.46, 0.04, 0.10, 0.06])
        self.x_slider = Slider(slider_x_ax, "target x", -self.reach, self.reach, valinit=self.target[0])
        self.y_slider = Slider(slider_y_ax, "target y", -self.reach, self.reach, valinit=self.target[1])
        self.reset_button = Button(reset_ax, "Reset")

        self.x_slider.on_changed(self.on_slider_changed)
        self.y_slider.on_changed(self.on_slider_changed)
        self.reset_button.on_clicked(self.on_reset)
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)

        self.target_artist = None
        self.status_artist = None
        self.recompute_samples()
        self.redraw()

    def recompute_samples(self):
        try:
            samples = jacobian_guided_sampling(
                self.arm,
                self.target,
                beta=self.beta,
                max_samples=self.max_samples,
                seed=self.seed,
            )
            residuals = np.linalg.norm(
                np.array([self.arm.forward_kinematics(q) - self.target for q in samples]),
                axis=1,
            )
            self.samples = samples
            self.status = (
                f"target=({self.target[0]:.2f}, {self.target[1]:.2f})  "
                f"samples={len(samples)}  max residual={residuals.max():.1e}"
            )
        except Exception as exc:
            self.samples = np.empty((0, 3))
            self.status = f"target=({self.target[0]:.2f}, {self.target[1]:.2f})  {exc}"

    def redraw(self):
        self.draw_arm_panel()
        self.draw_manifold_panel()
        self.fig.canvas.draw_idle()

    def draw_arm_panel(self):
        ax = self.ax_arm
        ax.clear()
        ax.set_xlim(-self.reach, self.reach)
        ax.set_ylim(-self.reach, self.reach)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.35)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Drag the target to update self-motion samples")

        if len(self.samples) > 0:
            idx = np.linspace(0, len(self.samples) - 1, min(14, len(self.samples))).astype(int)
            for draw_index, q in enumerate(self.samples[idx]):
                pts = self.arm.joint_positions(q)
                alpha = 0.25 + 0.5 * draw_index / max(1, len(idx) - 1)
                ax.plot(pts[:, 0], pts[:, 1], marker="o", linewidth=1.4, alpha=alpha)

        self.target_artist = ax.scatter(
            [self.target[0]],
            [self.target[1]],
            marker="x",
            s=130,
            linewidths=2.5,
            color="tab:red",
            label="target",
            zorder=5,
        )
        ax.legend(loc="upper right")
        self.status_artist = ax.text(
            0.02,
            0.02,
            self.status,
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.75, "edgecolor": "0.8"},
        )

    def draw_manifold_panel(self):
        ax = self.ax_manifold
        ax.clear()
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_zlim(-np.pi, np.pi)
        ax.set_xlabel("q1 [rad]")
        ax.set_ylabel("q2 [rad]")
        ax.set_zlabel("q3 [rad]")
        ax.set_title("Null-space manifold samples")

        if len(self.samples) > 0:
            ax.plot(
                self.samples[:, 0],
                self.samples[:, 1],
                self.samples[:, 2],
                marker="o",
                markersize=3,
                linewidth=1.2,
            )
            ax.scatter(
                [self.samples[0, 0]],
                [self.samples[0, 1]],
                [self.samples[0, 2]],
                s=55,
                color="tab:red",
                label="initial IK",
            )
            ax.legend(loc="upper left")

    def set_target(self, x, y, update_sliders=True):
        r = np.hypot(x, y)
        if r > self.reach:
            x *= self.reach / r
            y *= self.reach / r
        self.target = np.array([x, y], dtype=float)
        if update_sliders:
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

    def update_target_and_redraw(self, x, y, force=False):
        now = time.monotonic()
        if not force and now - self.last_update_time < self.update_interval:
            return
        self.last_update_time = now
        self.set_target(x, y, update_sliders=False)
        self.recompute_samples()
        self.redraw()

    def on_press(self, event):
        if event.inaxes != self.ax_arm or event.xdata is None or event.ydata is None:
            return
        if np.hypot(event.xdata - self.target[0], event.ydata - self.target[1]) < 0.25:
            self.dragging = True
            self.update_target_and_redraw(event.xdata, event.ydata, force=True)

    def on_motion(self, event):
        if not self.dragging or event.inaxes != self.ax_arm or event.xdata is None or event.ydata is None:
            return
        self.update_target_and_redraw(event.xdata, event.ydata)

    def on_release(self, event):
        if not self.dragging:
            return
        self.dragging = False
        if event.inaxes == self.ax_arm and event.xdata is not None and event.ydata is not None:
            self.update_target_and_redraw(event.xdata, event.ydata, force=True)
        else:
            self.recompute_samples()
            self.redraw()
        self.sync_sliders()

    def on_slider_changed(self, _):
        if self.dragging:
            return
        self.set_target(self.x_slider.val, self.y_slider.val, update_sliders=False)
        self.recompute_samples()
        self.redraw()

    def on_reset(self, _):
        self.set_target(1.3, 0.7, update_sliders=True)
        self.recompute_samples()
        self.redraw()

    def show(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Interactive null-space self-motion demo.")
    parser.add_argument("--x", type=float, default=1.3, help="initial target x position")
    parser.add_argument("--y", type=float, default=0.7, help="initial target y position")
    parser.add_argument("--beta", type=float, default=0.20, help="null-space tracing step size")
    parser.add_argument("--max-samples", type=int, default=180, help="maximum manifold samples per update")
    parser.add_argument("--seed", type=int, default=42, help="IK search seed")
    args = parser.parse_args()

    arm = PlanarArm3DOF(link_lengths=(1.0, 1.0, 0.7))
    demo = InteractiveSelfMotionDemo(
        arm,
        initial_target=(args.x, args.y),
        beta=args.beta,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    demo.show()


if __name__ == "__main__":
    main()
