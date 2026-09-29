"""Eigenvalue solvers: power iteration and a shifted QR algorithm for symmetric matrices."""
import numpy as np
from .qr import qr_decompose


def power_iteration(A, max_iter=10_000, tol=1e-12, seed=0):
    """Dominant eigenvalue/eigenvector of a square matrix (largest |lambda|)."""
    A = np.asarray(A, dtype=float)
    v = np.random.default_rng(seed).standard_normal(A.shape[0])
    v /= np.sqrt(v @ v)
    lam = 0.0
    for _ in range(max_iter):
        w = A @ v
        new_lam = v @ w                                # Rayleigh quotient
        v = w / np.sqrt(w @ w)
        if abs(new_lam - lam) < tol * max(1.0, abs(new_lam)):
            break
        lam = new_lam
    return float(new_lam), v


def eigh_qr(A, tol=1e-12, max_iter=10_000):
    """All eigenvalues and eigenvectors of a symmetric matrix.

    QR algorithm with Wilkinson shifts and deflation.
    Returns (eigenvalues sorted descending, eigenvectors as columns).
    """
    T = np.array(A, dtype=float)
    if T.ndim != 2 or T.shape[0] != T.shape[1] or not np.allclose(T, T.T):
        raise ValueError("A must be a symmetric square matrix")
    n = T.shape[0]
    V = np.eye(n)
    hi, it = n, 0
    while hi > 1:
        scale = (abs(T[hi - 1, hi - 1]) + abs(T[hi - 2, hi - 2])) or 1.0
        if np.max(np.abs(T[hi - 1, :hi - 1])) <= tol * scale:
            T[hi - 1, :hi - 1] = 0.0
            T[:hi - 1, hi - 1] = 0.0
            hi -= 1                                    # deflate: last eigenvalue converged
            continue
        if it >= max_iter:
            raise RuntimeError("QR algorithm did not converge")
        it += 1
        a, b, c = T[hi - 2, hi - 2], T[hi - 1, hi - 1], T[hi - 1, hi - 2]
        d = (a - b) / 2.0
        mu = b - (1.0 if d >= 0 else -1.0) * c * c / (abs(d) + np.sqrt(d * d + c * c))  # Wilkinson shift
        Q, R = qr_decompose(T[:hi, :hi] - mu * np.eye(hi))
        T[:hi, :hi] = R @ Q + mu * np.eye(hi)
        V[:, :hi] = V[:, :hi] @ Q
    vals = np.diag(T).copy()
    order = np.argsort(-vals)
    return vals[order], V[:, order]
