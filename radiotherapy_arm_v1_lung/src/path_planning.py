"""Independent NumPy RRT-Connect joint-space planner for the 6R arm.

Collision model: conservative spherical obstacle versus capsules whose axes join
consecutive joint pivots and the tool point. This is a *simplified planning model*,
not an exact STL collision check or a radiotherapy patient-safety model.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np


@dataclass(frozen=True)
class SphereObstacle:
    center: np.ndarray
    radius: float
    name: str = "planning_obstacle"

    def __post_init__(self):
        c = np.asarray(self.center, dtype=float)
        if c.shape != (3,) or not np.all(np.isfinite(c)) or self.radius <= 0:
            raise ValueError("Obstacle requires finite 3D centre and positive radius")
        object.__setattr__(self, "center", c)


@dataclass
class PlanResult:
    waypoints: np.ndarray
    raw_waypoints: np.ndarray
    elapsed_s: float
    collision_checks: int
    iterations: int
    success: bool


def point_segment_distance(point, start, end):
    """Euclidean distance from a point to a finite 3D line segment."""
    point, start, end = [np.asarray(p, dtype=float) for p in (point, start, end)]
    v = end - start
    scale = np.dot(v, v)
    t = np.clip(np.dot(point - start, v) / scale, 0.0, 1.0) if scale > 1e-16 else 0.0
    return float(np.linalg.norm(start + t * v - point))


class CollisionChecker:
    """Checks joint limits and simplified link-obstacle clearance.

    Requires an object with joint_limits (6,2) and _chain_state(q) returning
    (tool_point, joint_axes, joint_pivots), as in src.kinematics.RobotKinematics.
    Does not handle self-collisions, floor contacts, or full mesh geometries.
    """

    def __init__(self, kinematics, obstacles=(), link_radius=0.035, clearance=0.01,
                 interpolation_step=0.055):
        self.kinematics = kinematics
        self.obstacles = tuple(obstacles)
        self.link_radius = float(link_radius)
        self.clearance = float(clearance)
        self.interpolation_step = float(interpolation_step)
        if self.link_radius < 0 or self.clearance < 0 or self.interpolation_step <= 0:
            raise ValueError("Invalid collision geometry or interpolation step")
        self.checks = 0

    def valid(self, q):
        self.checks += 1
        q = np.asarray(q, dtype=float)
        limits = self.kinematics.joint_limits
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            return False
        if np.any(q < limits[:, 0] - 1e-10) or np.any(q > limits[:, 1] + 1e-10):
            return False
        if not self.obstacles:
            return True
        tip, _, pivots = self.kinematics._chain_state(q)
        points = np.vstack([pivots.T, tip])
        # Each link is approximated by a straight capsule; joint 1->2,...,6->tip.
        for obstacle in self.obstacles:
            separation = obstacle.radius + self.link_radius + self.clearance
            for a, b in zip(points[:-1], points[1:]):
                if point_segment_distance(obstacle.center, a, b) <= separation:
                    return False
        return True

    def edge_valid(self, a, b):
        a, b = np.asarray(a), np.asarray(b)
        n = max(1, int(np.ceil(np.max(np.abs(b-a)) / self.interpolation_step)))
        return all(self.valid(a + (b-a)*i/n) for i in range(n+1))


def _steer(a, b, step):
    d = b - a
    length = np.linalg.norm(d)
    if length < 1e-12:
        return a.copy()
    return a + d * min(1.0, step / length)


class _Tree:
    def __init__(self, root):
        self.nodes = [np.asarray(root, dtype=float).copy()]
        self.parents = [-1]

    def nearest(self, q):
        distances = np.linalg.norm(np.asarray(self.nodes) - q, axis=1)
        return int(np.argmin(distances))

    def add(self, q, parent):
        self.nodes.append(q.copy())
        self.parents.append(parent)
        return len(self.nodes)-1

    def path(self, idx):
        path = []
        while idx >= 0:
            path.append(self.nodes[idx])
            idx = self.parents[idx]
        return list(reversed(path))


def _extend(tree, q_target, checker, step):
    i = tree.nearest(q_target)
    nxt = _steer(tree.nodes[i], q_target, step)
    if np.linalg.norm(nxt-tree.nodes[i]) < 1e-10 or not checker.edge_valid(tree.nodes[i], nxt):
        return "trapped", i
    idx = tree.add(nxt, i)
    return ("reached" if np.linalg.norm(nxt-q_target)<1e-8 else "advanced"), idx


def _connect(tree, target, checker, step):
    status = "advanced"
    index = 0
    while status == "advanced":
        status, index = _extend(tree, target, checker, step)
    return status, index


def shortcut_path(path, checker, rng, attempts=100):
    pts = [p.copy() for p in path]
    for _ in range(attempts):
        if len(pts) <= 2:
            break
        i, j = sorted(rng.choice(len(pts), size=2, replace=False))
        if j <= i+1:
            continue
        if checker.edge_valid(pts[i], pts[j]):
            pts = pts[:i+1] + pts[j:]
    return np.asarray(pts)


def rrt_connect(start, goal, checker, *, step=0.18, max_iterations=4000,
                goal_bias=0.15, seed=7, smooth_attempts=120):
    """Bidirectional RRT-Connect (Kuffner/LaValle style); returns PlanResult.

    All tree construction, nearest neighbour and steering are authored in NumPy.
    """
    begin = time.perf_counter()
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    if not checker.valid(start):
        raise ValueError("Start configuration violates joint limits or obstacle clearance")
    if not checker.valid(goal):
        raise ValueError("Goal configuration violates joint limits or obstacle clearance")
    if checker.edge_valid(start, goal):
        path = np.vstack([start, goal])
        return PlanResult(path, path.copy(), time.perf_counter()-begin, checker.checks, 0, True)
    rng = np.random.default_rng(seed)
    lower, upper = checker.kinematics.joint_limits.T
    trees = [_Tree(start), _Tree(goal)]
    for iteration in range(1, max_iterations+1):
        a = (iteration-1)%2
        b = 1-a
        sample = trees[b].nodes[0] if rng.random() < goal_bias else rng.uniform(lower, upper)
        status, i = _extend(trees[a], sample, checker, step)
        if status == "trapped":
            continue
        status2, j = _connect(trees[b], trees[a].nodes[i], checker, step)
        if status2 != "reached":
            continue
        if a == 0:
            combined = trees[0].path(i) + list(reversed(trees[1].path(j)))[1:]
        else:
            combined = trees[0].path(j) + list(reversed(trees[1].path(i)))[1:]
        raw = np.asarray(combined)
        smooth = shortcut_path(raw, checker, rng, smooth_attempts)
        return PlanResult(smooth, raw, time.perf_counter()-begin,
                          checker.checks, iteration, True)
    return PlanResult(np.empty((0,6)), np.empty((0,6)), time.perf_counter()-begin,
                      checker.checks, max_iterations, False)
