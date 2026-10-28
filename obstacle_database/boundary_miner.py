import numpy as np


class BoundaryMiner:
    def __init__(self, tolerance=1e-4, max_iterations=40):
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)

    def find_boundary(self, q_free, q_collision, collision_fn):
        q_free = np.asarray(q_free, dtype=float)
        q_collision = np.asarray(q_collision, dtype=float)
        lo = q_free.copy()
        hi = q_collision.copy()
        for _ in range(self.max_iterations):
            if np.linalg.norm(hi - lo) <= self.tolerance:
                break
            mid = 0.5 * (lo + hi)
            if collision_fn(mid):
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def signed_distance_and_gradient(self, q, q_boundary, collision_label):
        q = np.asarray(q, dtype=float)
        q_boundary = np.asarray(q_boundary, dtype=float)
        diff = q - q_boundary
        norm = float(np.linalg.norm(diff))
        sign = -1.0 if collision_label else 1.0
        if norm <= 1e-12:
            return 0.0, np.zeros_like(q)
        return sign * norm, sign * diff / norm


def nearest_opposite_pairs(q, labels, max_pairs=2000):
    labels = np.asarray(labels).astype(bool)
    free_ids = np.flatnonzero(~labels)
    collision_ids = np.flatnonzero(labels)
    if len(free_ids) == 0 or len(collision_ids) == 0:
        return []
    pairs = []
    if len(free_ids) <= len(collision_ids):
        source_ids = free_ids[:max_pairs]
        target_ids = collision_ids
        source_is_free = True
    else:
        source_ids = collision_ids[:max_pairs]
        target_ids = free_ids
        source_is_free = False
    target_q = q[target_ids]
    for sid in source_ids:
        d = np.linalg.norm(target_q - q[sid], axis=1)
        tid = int(target_ids[int(np.argmin(d))])
        pairs.append((int(sid), tid) if source_is_free else (tid, int(sid)))
    return pairs
