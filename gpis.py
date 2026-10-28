import numpy as np
from planar_arm import PlanarArm3DOF


class GPIS:
    """Shifted GPIS used as an implicit field around manifold samples.

    Training targets are y=1 at sample points. With an RBF kernel, phi(q)
    decays away from sampled manifold points and can be converted to a rough
    distance field by inverting the Gaussian kernel.
    """

    def __init__(self, length_scale=0.35, noise=1e-5):
        self.length_scale = float(length_scale)
        self.noise = float(noise)
        self.samples = None
        self.alpha = None

    def _diff(self, A, B):
        # Respect joint periodicity in [-pi, pi].
        return PlanarArm3DOF.wrap_to_pi(A[:, None, :] - B[None, :, :])

    def kernel(self, A, B):
        A = np.atleast_2d(A).astype(float)
        B = np.atleast_2d(B).astype(float)
        diff = self._diff(A, B)
        sqdist = np.sum(diff * diff, axis=-1)
        return np.exp(-sqdist / (2.0 * self.length_scale ** 2))

    def fit(self, samples):
        self.samples = np.asarray(samples, dtype=float)
        y = np.ones(len(self.samples))
        K = self.kernel(self.samples, self.samples)
        K += (self.noise ** 2) * np.eye(len(self.samples))
        self.alpha = np.linalg.solve(K, y)
        return self

    def phi(self, q):
        q2 = np.atleast_2d(q).astype(float)
        Kq = self.kernel(self.samples, q2)
        out = Kq.T @ self.alpha
        return out if np.ndim(q) > 1 else float(out[0])

    def grad_phi(self, q):
        q = np.asarray(q, dtype=float)
        diff = PlanarArm3DOF.wrap_to_pi(q[None, :] - self.samples)
        k = np.exp(-np.sum(diff * diff, axis=1) / (2.0 * self.length_scale ** 2))
        # d/dq exp(-||q-s||^2/2l^2) = -(q-s)/l^2 * k
        grad = np.sum((self.alpha * k)[:, None] * (-(diff) / (self.length_scale ** 2)), axis=0)
        return grad

    def distance(self, q):
        phi = np.clip(self.phi(q), 1e-12, 1.0)
        return np.sqrt(np.maximum(0.0, -2.0 * self.length_scale ** 2 * np.log(phi)))

    def project_to_manifold(self, q, steps=5, step_scale=1.0):
        """Move q toward the implicit manifold using GPIS distance and gradient."""
        q = np.asarray(q, dtype=float).copy()
        trajectory = [q.copy()]
        for _ in range(steps):
            g = self.grad_phi(q)
            ng = np.linalg.norm(g)
            if ng < 1e-12:
                break
            d = self.distance(q)
            q = PlanarArm3DOF.wrap_to_pi(q + step_scale * d * g / ng)
            trajectory.append(q.copy())
            if d < 1e-3:
                break
        return q, np.asarray(trajectory)
