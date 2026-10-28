import numpy as np
import matplotlib.pyplot as plt


def plot_manifold_3d(samples, output_path):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(samples[:, 0], samples[:, 1], samples[:, 2], marker="o", markersize=3)
    ax.set_xlabel("q1 [rad]")
    ax.set_ylabel("q2 [rad]")
    ax.set_zlabel("q3 [rad]")
    ax.set_title("Jacobian-guided null-space manifold samples")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_distance_slice(gpis, fixed_q3, output_path, resolution=160):
    q1 = np.linspace(-np.pi, np.pi, resolution)
    q2 = np.linspace(-np.pi, np.pi, resolution)
    Q1, Q2 = np.meshgrid(q1, q2)
    points = np.column_stack([Q1.ravel(), Q2.ravel(), np.full(Q1.size, fixed_q3)])
    phi = gpis.phi(points).reshape(Q1.shape)
    dist = np.array([gpis.distance(p) for p in points]).reshape(Q1.shape)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(
        dist,
        origin="lower",
        extent=[-np.pi, np.pi, -np.pi, np.pi],
        aspect="auto",
    )
    cs = ax.contour(Q1, Q2, phi, levels=[0.8, 0.9, 0.99], linewidths=1.0)
    ax.clabel(cs, inline=True, fontsize=8)
    ax.set_xlabel("q1 [rad]")
    ax.set_ylabel("q2 [rad]")
    ax.set_title(f"GPIS distance slice at q3={fixed_q3:.2f}")
    fig.colorbar(im, ax=ax, label="distance to manifold [rad]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_arm_motion(arm, qs, target, output_path):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    for q in qs:
        pts = arm.joint_positions(q)
        ax.plot(pts[:, 0], pts[:, 1], marker="o", alpha=0.45)
    ax.scatter([target[0]], [target[1]], marker="x", s=100, label="target")
    reach = np.sum(arm.link_lengths)
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend()
    ax.set_title("End-effector-preserving self-motion samples")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
