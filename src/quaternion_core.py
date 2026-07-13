"""Shared quaternion primitives — re-export from flux_hopf_lib.

``encode_decode.Quaternion`` historically lived as a local class. Prefer::

    from quaternion_core import Quaternion, rodrigues_rotation

or install flux_hopf_lib and import directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flux_hopf_lib.quaternion.core import Quaternion as _CoreQuaternion
from flux_hopf_lib.quaternion.core import rodrigues_rotation


@dataclass
class Quaternion(_CoreQuaternion):
    """Hamilton quaternion with legacy vqc_sims_public helpers."""

    def __repr__(self) -> str:
        return f"q({self.w:.3f} + {self.x:.3f}i + {self.y:.3f}j + {self.z:.3f}k)"

    def normalize(self) -> Quaternion:
        n = self.norm()
        if n < 1e-12:
            return type(self)(1.0, 0.0, 0.0, 0.0)
        return type(self)(self.w / n, self.x / n, self.y / n, self.z / n)

    def conjugate(self) -> Quaternion:
        return type(self)(self.w, -self.x, -self.y, -self.z)

    def inverse(self) -> Quaternion:
        n2 = self.norm() ** 2
        if n2 < 1e-16:
            raise ZeroDivisionError("cannot invert near-zero quaternion")
        return type(self)(self.w / n2, -self.x / n2, -self.y / n2, -self.z / n2)

    def multiply(self, other: _CoreQuaternion) -> Quaternion:
        return type(self)(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    def rotate_vector(self, v: np.ndarray) -> np.ndarray:
        """Rotate 3-vector ``v`` by this unit quaternion (sandwich product)."""
        v = np.asarray(v, dtype=float).reshape(3)
        qv = type(self)(0.0, float(v[0]), float(v[1]), float(v[2]))
        rotated = self.multiply(qv).multiply(self.inverse())
        return np.array([rotated.x, rotated.y, rotated.z], dtype=float)


def quaternion_encode(data: np.ndarray) -> Quaternion:
    """Map a data shard to a unit quaternion (compression proxy)."""
    arr = np.asarray(data, dtype=float).flatten()
    if arr.size < 4:
        arr = np.pad(arr, (0, 4 - arr.size))
    norm = np.linalg.norm(arr[:4]) + 1e-12
    vec = arr[:4] / norm
    return Quaternion(float(vec[0]), float(vec[1]), float(vec[2]), float(vec[3]))


__all__ = [
    "Quaternion",
    "rodrigues_rotation",
    "quaternion_encode",
]
