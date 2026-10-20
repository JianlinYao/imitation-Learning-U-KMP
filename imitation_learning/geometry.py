"""Continuous polyline-circle clearance and a common geometric via-point policy."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Circle:
    center: np.ndarray
    radius: float
    margin: float

    def __post_init__(self):
        center = np.asarray(self.center, dtype=float)
        if center.shape != (2,) or not np.isfinite(center).all() or self.radius <= 0 or self.margin < 0:
            raise ValueError("invalid circle geometry")
        object.__setattr__(self, "center", center.copy())

    def as_dict(self):
        return {"center": self.center.tolist(), "radius": float(self.radius), "margin": float(self.margin)}


def segment_clearances(path: np.ndarray, circle: Circle) -> np.ndarray:
    """Signed distance from each entire segment to the obstacle boundary."""
    start, end = path[:-1], path[1:]
    direction = end - start
    square = np.sum(direction ** 2, axis=1)
    projection = np.sum((circle.center - start) * direction, axis=1) / np.maximum(square, 1e-30)
    projection = np.clip(projection, 0, 1)
    closest = start + projection[:, None] * direction
    return np.linalg.norm(closest - circle.center, axis=1) - circle.radius


def clearance_metrics(path: np.ndarray, circle: Circle) -> dict:
    clearance = float(segment_clearances(path, circle).min())
    return {"collision": bool(clearance <= 0), "min_clearance": clearance,
            "margin_satisfied": bool(clearance >= circle.margin)}


def make_obstacle(reference: np.ndarray, radius_fraction: float = 0.07,
                  margin_fraction: float = 0.02) -> Circle:
    span = float(np.linalg.norm(np.ptp(reference, axis=0)))
    if span < 1e-8 or radius_fraction <= 0 or margin_fraction < 0:
        raise ValueError("obstacle requires a non-degenerate reference and valid fractions")
    candidates = np.arange(int(0.3 * len(reference)), int(0.7 * len(reference)))
    distances = np.minimum(np.linalg.norm(reference[candidates] - reference[0], axis=1),
                           np.linalg.norm(reference[candidates] - reference[-1], axis=1))
    index = candidates[np.argmax(distances)]
    radius = min(radius_fraction * span, float(distances.max()) / 3.5)
    return Circle(reference[index], radius, margin_fraction * span)


def detour_points(reference: np.ndarray, circle: Circle) -> list[tuple[int, np.ndarray]]:
    """Project a shared reference locally around a circle; no test interior used.

    This supplies constraints, not a guarantee of collision avoidance. Every
    generated candidate must still pass continuous segment clearance checks.
    """
    distance = np.linalg.norm(reference - circle.center, axis=1)
    safe_radius = circle.radius + circle.margin
    unsafe = np.flatnonzero(distance < safe_radius * 1.6)
    if len(unsafe) == 0:
        return []
    via = []
    groups = np.split(unsafe, np.flatnonzero(np.diff(unsafe) > 1) + 1)
    # Multiple visits to the same obstacle are distinct local adaptations. Do
    # not flatten the distant loop between first and last contact onto one side.
    padding = max(3, int(0.04 * len(reference)))
    for group in groups:
        lo = max(1, int(group[0]) - padding)
        hi = min(len(reference) - 2, int(group[-1]) + padding)
        tangent = reference[hi] - reference[lo]
        normal = np.array([-tangent[1], tangent[0]])
        normal /= max(np.linalg.norm(normal), 1e-12)
        if np.linalg.norm(normal) < 0.5:
            normal = np.array([0.0, 1.0])
        midpoint = (reference[lo] + reference[hi]) * 0.5 - circle.center
        if np.dot(midpoint, normal) < 0:
            normal *= -1
        for index in np.unique(np.linspace(lo, hi, 5, dtype=int)):
            vector = reference[index] - circle.center
            if np.linalg.norm(vector) >= safe_radius * 1.35:
                projected = reference[index].copy()
            else:
                longitudinal = vector - normal * np.dot(vector, normal)
                height = np.sqrt(max((safe_radius * 1.35) ** 2 - np.dot(longitudinal, longitudinal), 0.0))
                projected = circle.center + longitudinal + normal * max(height, np.dot(vector, normal))
            via.append((int(index), projected))
    return via


def repair_point(path: np.ndarray, circle: Circle) -> tuple[int, np.ndarray]:
    clearances = segment_clearances(path, circle)
    segment = int(np.argmin(clearances))
    index = min(max(segment + 1, 1), len(path) - 2)
    vector = path[index] - circle.center
    norm = np.linalg.norm(vector)
    if norm < 1e-10:
        tangent = path[min(index + 2, len(path) - 1)] - path[max(0, index - 2)]
        vector = np.array([-tangent[1], tangent[0]])
        norm = np.linalg.norm(vector)
    if norm < 1e-10:
        vector, norm = np.array([0.0, 1.0]), 1.0
    target = circle.center + vector / norm * (circle.radius + 1.7 * circle.margin)
    return index, target
