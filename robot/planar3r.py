import numpy as np


def point_segment_distance(point, a, b):
    """Euclidean distance to a closed segment, including degenerate segments."""
    point = np.asarray(point, dtype=float)
    closest = closest_point_on_segment(point, a, b)
    return float(np.linalg.norm(point - closest))


def closest_point_on_segment(point, a, b):
    point = np.asarray(point, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-15:
        return a.copy()
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    return a + t * ab


def orientation(a, b, c):
    ab, ac = b - a, c - a
    return float(ab[0] * ac[1] - ab[1] * ac[0])


def segments_intersect(a, b, c, d):
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    return o1 * o2 < 0.0 and o3 * o4 < 0.0


def segment_segment_distance(a, b, c, d):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)
    if segments_intersect(a, b, c, d):
        return 0.0
    return min(
        point_segment_distance(a, c, d),
        point_segment_distance(b, c, d),
        point_segment_distance(c, a, b),
        point_segment_distance(d, a, b),
    )


class Planar3RRobot:
    """Planar 3R robot with simple circle/capsule collision geometry."""

    def __init__(
        self,
        link_lengths=(1.0, 1.0, 0.7),
        joint_limits=None,
        link_radius=0.06,
        circle_radius=0.06,
        circle_spacing=0.08,
    ):
        self.link_lengths = np.asarray(link_lengths, dtype=float)
        if self.link_lengths.shape != (3,):
            raise ValueError("link_lengths must contain 3 values")
        if joint_limits is None:
            joint_limits = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
        self.joint_limits = np.asarray(joint_limits, dtype=float)
        if self.joint_limits.shape != (3, 2):
            raise ValueError("joint_limits must have shape (3, 2)")
        self.link_radius = float(link_radius)
        self.circle_radius = float(circle_radius)
        self.circle_spacing = float(circle_spacing)
        self._circle_t_values = [
            np.linspace(0.0, 1.0, max(2, int(np.ceil(length / self.circle_spacing)) + 1))
            for length in self.link_lengths
        ]
        self.num_circles = int(sum(len(t) for t in self._circle_t_values))

    def sample_configurations(self, count, seed=42, dtype=np.float32):
        rng = np.random.default_rng(seed)
        low = self.joint_limits[:, 0]
        high = self.joint_limits[:, 1]
        return rng.uniform(low, high, size=(count, 3)).astype(dtype)

    def forward_kinematics(self, q):
        q = np.asarray(q, dtype=float)
        l1, l2, l3 = self.link_lengths
        p0 = np.array([0.0, 0.0])
        t1 = q[0]
        p1 = p0 + l1 * np.array([np.cos(t1), np.sin(t1)])
        t2 = q[0] + q[1]
        p2 = p1 + l2 * np.array([np.cos(t2), np.sin(t2)])
        t3 = q[0] + q[1] + q[2]
        p3 = p2 + l3 * np.array([np.cos(t3), np.sin(t3)])
        return np.vstack([p0, p1, p2, p3])

    def link_segments(self, q):
        pts = self.forward_kinematics(q)
        return [(pts[i], pts[i + 1]) for i in range(3)]

    def collision_circles(self, q):
        pts = self.forward_kinematics(q)
        centers = []
        radii = []
        for link_index, t_values in enumerate(self._circle_t_values):
            a = pts[link_index]
            b = pts[link_index + 1]
            for t in t_values:
                centers.append((1.0 - t) * a + t * b)
                radii.append(self.circle_radius)
        return np.asarray(centers, dtype=np.float32), np.asarray(radii, dtype=np.float32)

    def collision_circles_batch(self, q_values):
        centers = np.empty((len(q_values), self.num_circles, 2), dtype=np.float32)
        radii = None
        for i, q in enumerate(q_values):
            c, r = self.collision_circles(q)
            centers[i] = c
            if radii is None:
                radii = r
        return centers, radii

    def self_collision(self, q):
        pts = self.forward_kinematics(q)
        d13 = segment_segment_distance(pts[0], pts[1], pts[2], pts[3])
        return bool(d13 <= 2.0 * self.link_radius)

    def self_collision_batch(self, q_values):
        return np.asarray([self.self_collision(q) for q in q_values], dtype=np.uint8)

    def external_collision(self, q, obstacle_center, obstacle_radius):
        obstacle_center = np.asarray(obstacle_center, dtype=float)
        threshold = self.link_radius + float(obstacle_radius)
        for a, b in self.link_segments(q):
            if point_segment_distance(obstacle_center, a, b) <= threshold:
                return True
        return False

    def link_point_distance(self, q, point):
        point = np.asarray(point, dtype=float)
        return min(point_segment_distance(point, a, b) for a, b in self.link_segments(q))

    def closest_link_point(self, q, point):
        point = np.asarray(point, dtype=float)
        best_point = None
        best_distance = np.inf
        best_link_index = -1
        for link_index, (a, b) in enumerate(self.link_segments(q)):
            closest = closest_point_on_segment(point, a, b)
            distance = float(np.linalg.norm(point - closest))
            if distance < best_distance:
                best_distance = distance
                best_point = closest
                best_link_index = link_index
        return best_point, best_distance, best_link_index

    def workspace_bounds(self, margin=0.0):
        reach = float(np.sum(self.link_lengths)) + float(margin)
        return np.array([-reach, -reach], dtype=float), np.array([reach, reach], dtype=float)
