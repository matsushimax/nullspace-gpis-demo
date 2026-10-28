import os
import numpy as np

from planar_arm import PlanarArm3DOF
from sampler import jacobian_guided_sampling
from gpis import GPIS
from visualize import plot_manifold_3d, plot_distance_slice, plot_arm_motion


def main():
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    arm = PlanarArm3DOF(link_lengths=(1.0, 1.0, 0.7))
    target = np.array([1.3, 0.7])

    samples = jacobian_guided_sampling(
        arm,
        target,
        beta=0.20,
        max_samples=220,
        seed=42,
    )
    print(f"samples: {len(samples)}")
    residuals = np.linalg.norm(np.array([arm.forward_kinematics(q) - target for q in samples]), axis=1)
    print(f"mean residual: {residuals.mean():.3e}, max residual: {residuals.max():.3e}")

    gpis = GPIS(length_scale=0.35, noise=1e-4).fit(samples)

    # Test GPIS projection from a random off-manifold posture.
    q_test = np.array([1.8, -1.7, 1.1])
    q_proj, traj = gpis.project_to_manifold(q_test, steps=8, step_scale=0.8)
    print("q_test:", q_test)
    print("q_proj:", q_proj)
    print("phi(q_test), distance:", gpis.phi(q_test), gpis.distance(q_test))
    print("phi(q_proj), distance:", gpis.phi(q_proj), gpis.distance(q_proj))

    np.save(os.path.join(out_dir, "samples.npy"), samples)
    np.save(os.path.join(out_dir, "projection_trajectory.npy"), traj)

    plot_manifold_3d(samples, os.path.join(out_dir, "manifold_samples_3d.png"))
    plot_distance_slice(gpis, fixed_q3=float(np.median(samples[:, 2])), output_path=os.path.join(out_dir, "gpis_distance_slice.png"))
    # Show a sparse set of robot postures along the manifold.
    idx = np.linspace(0, len(samples) - 1, min(12, len(samples))).astype(int)
    plot_arm_motion(arm, samples[idx], target, os.path.join(out_dir, "arm_self_motion.png"))

    print(f"saved results to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
