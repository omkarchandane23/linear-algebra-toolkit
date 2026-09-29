"""QR decomposition using Householder reflections, plus least squares."""
import numpy as np
from .lu import back_substitution


def qr_decompose(A):
    """Return (Q, R) with A = Q @ R, Q orthogonal (m x m), R upper triangular (m x n)."""
    R = np.array(A, dtype=float)
    m, n = R.shape
    Q = np.eye(m)
    for k in range(min(m - 1, n)):
        x = R[k:, k]
        norm = np.sqrt(x @ x)
        if norm == 0:
            continue
        v = x.copy()
        v[0] += (1.0 if x[0] >= 0 else -1.0) * norm    # sign choice avoids cancellation
        v /= np.sqrt(v @ v)
        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:])  # apply H = I - 2vv^T from the left
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ v, v)    # accumulate Q = H1 H2 ...
    return Q, np.triu(R)


def least_squares(A, b):
    """Minimize ||A x - b|| for a tall, full-column-rank A using QR."""
    A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
    n = A.shape[1]
    Q, R = qr_decompose(A)
    return back_substitution(R[:n, :n], (Q.T @ b)[:n])
