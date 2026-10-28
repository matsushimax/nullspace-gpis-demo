import numpy as np
from planar_arm import PlanarArm3DOF


def damped_pseudo_inverse(J, damping=1e-6):
    """Right damped pseudo inverse J^T (J J^T + lambda^2 I)^-1."""
    J = np.asarray(J, dtype=float)
    m = J.shape[0]
    return J.T @ np.linalg.inv(J @ J.T + (damping ** 2) * np.eye(m))


def gauss_newton_project(arm, q, target, tol=1e-8, max_iter=50, damping=1e-5):
    """Project q to f(q)=target using damped Gauss-Newton."""
    q = PlanarArm3DOF.wrap_to_pi(np.asarray(q, dtype=float).copy())
    target = np.asarray(target, dtype=float)

    for _ in range(max_iter):
        r = arm.forward_kinematics(q) - target
        if np.linalg.norm(r) <= tol:
            return q, True, np.linalg.norm(r)
        J = arm.jacobian(q)
        dq = -damped_pseudo_inverse(J, damping=damping) @ r
        # Prevent extreme jumps near singularities.
        step_norm = np.linalg.norm(dq)
        if step_norm > 0.5:
            dq *= 0.5 / step_norm
        q = PlanarArm3DOF.wrap_to_pi(q + dq)

    r = arm.forward_kinematics(q) - target
    return q, np.linalg.norm(r) <= tol * 10, np.linalg.norm(r)


def nullspace_basis(J):
    """Return an orthonormal basis for Null(J) using SVD."""
    _, s, vh = np.linalg.svd(J, full_matrices=True)
    rank = np.sum(s > 1e-8)
    return vh[rank:].T


def find_initial_solution(arm, target, seed=0, attempts=200):
    rng = np.random.default_rng(seed)
    # Try a few deterministic guesses first, then random guesses.
    guesses = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 0.5, -0.5]),
        np.array([1.0, -1.0, 1.0]),
        np.array([-1.0, 1.0, -1.0]),
    ]
    guesses += [rng.uniform(-np.pi, np.pi, size=3) for _ in range(attempts)]

    best = None
    for q0 in guesses:
        q, ok, residual = gauss_newton_project(arm, q0, target)
        if best is None or residual < best[2]:
            best = (q, ok, residual)
        if ok:
            return q
    raise RuntimeError(f"Could not find IK solution; best residual={best[2]:.3e}")


def jacobian_guided_sampling(
    arm,
    target,
    beta=0.25,
    max_samples=300,
    close_radius=None,
    seed=0,
):
    """Trace a 1D null-space manifold by tangent stepping + projection.

    This demo targets 3DOF arm / 2D task, so Null(J) is 1D away from singularities.
    """
    if close_radius is None:
        close_radius = beta * 0.75

    q0 = find_initial_solution(arm, target, seed=seed)
    samples = [q0]
    q = q0.copy()
    prev_v = None

    for k in range(max_samples - 1):
        J = arm.jacobian(q)
        N = nullspace_basis(J)
        if N.shape[1] < 1:
            break
        v = N[:, 0]
        v /= np.linalg.norm(v)

        # Keep tangent orientation continuous.
        if prev_v is not None and np.dot(v, prev_v) < 0:
            v = -v
        prev_v = v

        q_tilde = PlanarArm3DOF.wrap_to_pi(q + beta * v)
        q_next, ok, residual = gauss_newton_project(arm, q_tilde, target)
        if not ok:
            print(f"warning: projection residual={residual:.3e} at step {k}")

        # Stop when we return close to the initial sample after enough progress.
        if len(samples) > 10 and angular_distance(q_next, q0) < close_radius:
            break

        # Avoid stalling on an already visited point.
        if min(angular_distance(q_next, s) for s in samples) < 1e-4:
            break

        samples.append(q_next)
        q = q_next

    return np.asarray(samples)


def angular_distance(a, b):
    d = PlanarArm3DOF.wrap_to_pi(np.asarray(a) - np.asarray(b))
    return float(np.linalg.norm(d))
