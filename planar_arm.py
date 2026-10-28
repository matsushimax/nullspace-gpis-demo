import numpy as np


class PlanarArm3DOF:
    """3-DOF planar serial manipulator.

    q = [theta1, theta2, theta3]
    task = end-effector 2D position [x, y]
    """

    def __init__(self, link_lengths=(1.0, 1.0, 0.7)):
        self.link_lengths = np.asarray(link_lengths, dtype=float)
        if self.link_lengths.shape != (3,):
            raise ValueError("link_lengths must contain 3 values")

    @staticmethod
    def wrap_to_pi(q):
        return (q + np.pi) % (2.0 * np.pi) - np.pi

    def forward_kinematics(self, q):
        q = np.asarray(q, dtype=float)
        l1, l2, l3 = self.link_lengths
        t1 = q[0]
        t2 = q[0] + q[1]
        t3 = q[0] + q[1] + q[2]
        x = l1 * np.cos(t1) + l2 * np.cos(t2) + l3 * np.cos(t3)
        y = l1 * np.sin(t1) + l2 * np.sin(t2) + l3 * np.sin(t3)
        return np.array([x, y])

    def joint_positions(self, q):
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

    def jacobian(self, q):
        q = np.asarray(q, dtype=float)
        l1, l2, l3 = self.link_lengths
        t1 = q[0]
        t2 = q[0] + q[1]
        t3 = q[0] + q[1] + q[2]

        dx_dq1 = -l1*np.sin(t1) - l2*np.sin(t2) - l3*np.sin(t3)
        dx_dq2 = -l2*np.sin(t2) - l3*np.sin(t3)
        dx_dq3 = -l3*np.sin(t3)

        dy_dq1 = l1*np.cos(t1) + l2*np.cos(t2) + l3*np.cos(t3)
        dy_dq2 = l2*np.cos(t2) + l3*np.cos(t3)
        dy_dq3 = l3*np.cos(t3)

        return np.array([
            [dx_dq1, dx_dq2, dx_dq3],
            [dy_dq1, dy_dq2, dy_dq3],
        ])
