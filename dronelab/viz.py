"""Helpers shared by the HTML viewers: frame conversion and page writing.

The simulations use a z-up frame (x, y, z); three.js is y-up. Mapping sim
(x, y, z) to three (x, z, -y) keeps the frame right-handed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

P = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])   # sim -> three.js


def to_three(v: np.ndarray) -> np.ndarray:
    """Convert sim-frame vectors (..., 3) to the three.js frame."""
    return v @ P.T


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Rotation matrices (..., 3, 3) -> quaternions (..., 4) as (x, y, z, w)."""
    m = R.reshape(-1, 3, 3)
    q = np.zeros((len(m), 4))
    tr = np.trace(m, axis1=1, axis2=2)
    for k, M in enumerate(m):
        if tr[k] > 0:
            s = 2 * np.sqrt(tr[k] + 1.0)
            q[k] = [(M[2, 1] - M[1, 2]) / s, (M[0, 2] - M[2, 0]) / s, (M[1, 0] - M[0, 1]) / s, s / 4]
        else:
            i = int(np.argmax(np.diag(M)))
            j, l = (i + 1) % 3, (i + 2) % 3
            s = 2 * np.sqrt(1.0 + M[i, i] - M[j, j] - M[l, l])
            qq = np.zeros(4)
            qq[i] = s / 4
            qq[j] = (M[j, i] + M[i, j]) / s
            qq[l] = (M[l, i] + M[i, l]) / s
            qq[3] = (M[l, j] - M[j, l]) / s
            q[k] = qq
    q *= np.sign(q[:, 3:4] + 1e-12)
    return q.reshape(*R.shape[:-2], 4)


def rounded(a, nd: int = 3) -> list:
    """Flatten and round an array for compact JSON."""
    return np.round(np.asarray(a, dtype=np.float64).reshape(-1), nd).tolist()


def write_page(template: Path, data: dict, path: str | Path) -> Path:
    """Embed ``data`` as JSON into an HTML template and write it to ``path``."""
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = Path(template).read_text(encoding="utf-8").replace("/*__DATA__*/null", payload, 1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
