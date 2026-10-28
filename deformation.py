from dataclasses import dataclass

import numpy as np

from planar_arm import PlanarArm3DOF
from sampler import gauss_newton_project


@dataclass(slots=True)
class AnchorProjectionResult:
    source_indices: np.ndarray
    source_anchors: np.ndarray
    projected_anchors: np.ndarray
    converged: np.ndarray
    residual_norms: np.ndarray


@dataclass(slots=True)
class DeformationResult:
    estimated_q: np.ndarray
    anchor_result: AnchorProjectionResult
    sigma: float
    task_residuals: np.ndarray
    method: str = "rbf"
    post_projected: bool = False
    post_projection_converged: np.ndarray | None = None


def configuration_difference(a, b):
    return PlanarArm3DOF.wrap_to_pi(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


def pairwise_wrapped_distances(A, B):
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    diff = configuration_difference(A[:, None, :], B[None, :, :])
    return np.linalg.norm(diff, axis=-1)


def select_uniform_arc_length_anchors(manifold, num_anchors):
    count = len(manifold.q)
    if count == 0:
        return np.empty((0,), dtype=np.int32)
    num_anchors = max(1, min(int(num_anchors), count))
    arc = np.asarray(manifold.arc_length, dtype=float)
    if len(arc) != count or arc[-1] <= 1e-12:
        return np.linspace(0, count - 1, num_anchors).round().astype(np.int32)

    target_arc = np.linspace(arc[0], arc[-1], num_anchors)
    indices = np.searchsorted(arc, target_arc)
    indices = np.clip(indices, 0, count - 1)
    return np.unique(indices).astype(np.int32)


def project_anchors_to_target(
    arm,
    manifold,
    target,
    num_anchors=16,
    tol=1e-8,
    max_iter=50,
    damping=1e-5,
):
    indices = select_uniform_arc_length_anchors(manifold, num_anchors)
    source = manifold.q[indices]
    projected = []
    converged = []
    residuals = []
    for q in source:
        q_projected, ok, residual = gauss_newton_project(
            arm,
            q,
            target,
            tol=tol,
            max_iter=max_iter,
            damping=damping,
        )
        projected.append(q_projected)
        converged.append(ok)
        residuals.append(residual)

    projected = np.asarray(projected, dtype=float).reshape((-1, arm.link_lengths.shape[0]))
    converged = np.asarray(converged, dtype=bool)
    residuals = np.asarray(residuals, dtype=float)
    return AnchorProjectionResult(
        source_indices=indices,
        source_anchors=source,
        projected_anchors=projected,
        converged=converged,
        residual_norms=residuals,
    )


class GaussianRBFDeformer:
    def __init__(self, sigma=None, regularization=1e-8):
        self.sigma = sigma
        self.regularization = float(regularization)
        self.source_anchors = None
        self.weights = None

    def fit(self, source_anchors, target_anchors):
        source_anchors = np.asarray(source_anchors, dtype=float)
        target_anchors = np.asarray(target_anchors, dtype=float)
        if len(source_anchors) == 0:
            raise ValueError("at least one anchor is required")
        if source_anchors.shape != target_anchors.shape:
            raise ValueError("source_anchors and target_anchors must have the same shape")

        self.source_anchors = source_anchors
        sigma = self.sigma if self.sigma is not None else estimate_rbf_sigma(source_anchors)
        self.sigma = max(float(sigma), 1e-6)
        displacement = configuration_difference(target_anchors, source_anchors)

        K = self._kernel(source_anchors, source_anchors)
        K += self.regularization * np.eye(len(source_anchors))
        try:
            self.weights = np.linalg.solve(K, displacement)
        except np.linalg.LinAlgError:
            self.weights = np.linalg.lstsq(K, displacement, rcond=None)[0]
        return self

    def transform(self, q):
        q = np.asarray(q, dtype=float)
        K = self._kernel(np.atleast_2d(q), self.source_anchors)
        displacement = K @ self.weights
        transformed = PlanarArm3DOF.wrap_to_pi(np.atleast_2d(q) + displacement)
        return transformed if q.ndim > 1 else transformed[0]

    def _kernel(self, A, B):
        distances = pairwise_wrapped_distances(A, B)
        return np.exp(-(distances * distances) / (2.0 * self.sigma * self.sigma))


class ThinPlateSplineDeformer:
    def __init__(self, regularization=1e-6):
        self.regularization = float(regularization)
        self.source_anchors = None
        self.weights = None
        self.affine = None

    def fit(self, source_anchors, target_anchors):
        source_anchors = np.asarray(source_anchors, dtype=float)
        target_anchors = np.asarray(target_anchors, dtype=float)
        if len(source_anchors) < source_anchors.shape[1] + 1:
            raise ValueError("TPS needs at least dof + 1 anchors")
        if source_anchors.shape != target_anchors.shape:
            raise ValueError("source_anchors and target_anchors must have the same shape")

        self.source_anchors = source_anchors
        displacement = configuration_difference(target_anchors, source_anchors)
        K = self._kernel(source_anchors, source_anchors)
        K += self.regularization * np.eye(len(source_anchors))
        P = np.column_stack([np.ones(len(source_anchors)), source_anchors])
        zeros = np.zeros((P.shape[1], P.shape[1]))
        system = np.block([[K, P], [P.T, zeros]])
        rhs = np.vstack([displacement, np.zeros((P.shape[1], displacement.shape[1]))])
        try:
            params = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            params = np.linalg.lstsq(system, rhs, rcond=None)[0]
        self.weights = params[: len(source_anchors)]
        self.affine = params[len(source_anchors) :]
        return self

    def transform(self, q):
        q = np.asarray(q, dtype=float)
        q2 = np.atleast_2d(q)
        K = self._kernel(q2, self.source_anchors)
        P = np.column_stack([np.ones(len(q2)), q2])
        displacement = K @ self.weights + P @ self.affine
        transformed = PlanarArm3DOF.wrap_to_pi(q2 + displacement)
        return transformed if q.ndim > 1 else transformed[0]

    def _kernel(self, A, B):
        r = pairwise_wrapped_distances(A, B)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = r * r * np.log(np.maximum(r, 1e-12))
        out[r <= 1e-12] = 0.0
        return out


def estimate_rbf_sigma(source_anchors):
    distances = pairwise_wrapped_distances(source_anchors, source_anchors)
    upper = distances[np.triu_indices(len(source_anchors), k=1)]
    upper = upper[upper > 1e-12]
    if len(upper) == 0:
        return 1.0
    return float(np.median(upper))


def post_project_samples(arm, q_samples, target, tol=1e-8, max_iter=20, damping=1e-5):
    projected = []
    converged = []
    residuals = []
    for q in q_samples:
        q_projected, ok, residual = gauss_newton_project(
            arm,
            q,
            target,
            tol=tol,
            max_iter=max_iter,
            damping=damping,
        )
        projected.append(q_projected)
        converged.append(ok)
        residuals.append(residual)
    return (
        np.asarray(projected, dtype=float),
        np.asarray(converged, dtype=bool),
        np.asarray(residuals, dtype=float),
    )


def interpolate_manifold_rbf(
    arm,
    source_manifold,
    target,
    num_anchors=16,
    sigma=None,
    regularization=1e-8,
    require_converged=False,
    post_project=False,
):
    anchor_result = project_anchors_to_target(
        arm,
        source_manifold,
        target,
        num_anchors=num_anchors,
    )
    mask = anchor_result.converged if require_converged else np.ones(len(anchor_result.converged), dtype=bool)
    if np.sum(mask) < 1:
        raise RuntimeError("no usable projected anchors")

    deformer = GaussianRBFDeformer(sigma=sigma, regularization=regularization).fit(
        anchor_result.source_anchors[mask],
        anchor_result.projected_anchors[mask],
    )
    estimated_q = deformer.transform(source_manifold.q)
    post_converged = None
    if post_project:
        estimated_q, post_converged, _ = post_project_samples(arm, estimated_q, target)
    task_residuals = np.linalg.norm(
        np.asarray([arm.forward_kinematics(q) - target for q in estimated_q]),
        axis=1,
    )
    return DeformationResult(
        estimated_q=estimated_q,
        anchor_result=anchor_result,
        sigma=deformer.sigma,
        task_residuals=task_residuals,
        method="rbf",
        post_projected=post_project,
        post_projection_converged=post_converged,
    )


def interpolate_manifold_tps(
    arm,
    source_manifold,
    target,
    num_anchors=16,
    regularization=1e-6,
    require_converged=False,
    post_project=False,
):
    anchor_result = project_anchors_to_target(
        arm,
        source_manifold,
        target,
        num_anchors=num_anchors,
    )
    mask = anchor_result.converged if require_converged else np.ones(len(anchor_result.converged), dtype=bool)
    if np.sum(mask) < arm.link_lengths.shape[0] + 1:
        raise RuntimeError("TPS needs at least dof + 1 usable projected anchors")

    deformer = ThinPlateSplineDeformer(regularization=regularization).fit(
        anchor_result.source_anchors[mask],
        anchor_result.projected_anchors[mask],
    )
    estimated_q = deformer.transform(source_manifold.q)
    post_converged = None
    if post_project:
        estimated_q, post_converged, _ = post_project_samples(arm, estimated_q, target)
    task_residuals = np.linalg.norm(
        np.asarray([arm.forward_kinematics(q) - target for q in estimated_q]),
        axis=1,
    )
    return DeformationResult(
        estimated_q=estimated_q,
        anchor_result=anchor_result,
        sigma=0.0,
        task_residuals=task_residuals,
        method="tps",
        post_projected=post_project,
        post_projection_converged=post_converged,
    )
