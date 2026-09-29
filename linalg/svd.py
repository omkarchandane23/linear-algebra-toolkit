"""Singular value decomposition via one-sided Jacobi rotations, plus low-rank approximation."""
import numpy as np


def svd(A, tol=1e-12, max_sweeps=60):
    """Thin SVD: returns (U, s, Vt) with A ~= U @ diag(s) @ Vt and s sorted descending."""
    A = np.array(A, dtype=float)
    m, n = A.shape
    if m < n:                                          # work on the tall orientation
        U, s, Vt = svd(A.T, tol, max_sweeps)
        return Vt.T, s, U.T
    U, V = A.copy(), np.eye(n)
    for _ in range(max_sweeps):
        off = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                alpha, beta, gamma = U[:, p] @ U[:, p], U[:, q] @ U[:, q], U[:, p] @ U[:, q]
                if gamma == 0.0 or alpha * beta == 0.0:
                    continue
                off = max(off, abs(gamma) / np.sqrt(alpha * beta))
                zeta = (beta - alpha) / (2.0 * gamma)
                t = (1.0 if zeta >= 0 else -1.0) / (abs(zeta) + np.sqrt(1.0 + zeta * zeta))
                c = 1.0 / np.sqrt(1.0 + t * t)
                s = c * t
                for M in (U, V):                       # rotate columns p and q
                    mp = M[:, p].copy()
                    M[:, p] = c * mp - s * M[:, q]
                    M[:, q] = s * mp + c * M[:, q]
        if off < tol:
            break
    sing = np.sqrt(np.sum(U * U, axis=0))
    order = np.argsort(-sing)
    sing, U, V = sing[order], U[:, order], V[:, order]
    keep = sing > 1e-14 * max(sing[0], 1e-300)
    U[:, keep] /= sing[keep]
    U[:, ~keep] = 0.0
    return U, sing, V.T


def low_rank(A, k):
    """Best rank-k approximation of A (Eckart-Young theorem)."""
    U, s, Vt = svd(A)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k]
