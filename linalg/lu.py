"""LU decomposition with partial pivoting, plus linear-system solving."""
import numpy as np


def lu_decompose(A):
    """Return (P, L, U) such that P @ A = L @ U for a square matrix A."""
    U = np.array(A, dtype=float)
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("A must be a square matrix")
    n = U.shape[0]
    L, P = np.eye(n), np.eye(n)
    for k in range(n - 1):
        p = k + int(np.argmax(np.abs(U[k:, k])))      # partial pivoting
        if U[p, k] == 0:
            continue                                   # nothing to eliminate in this column
        if p != k:
            U[[k, p]] = U[[p, k]]
            P[[k, p]] = P[[p, k]]
            L[[k, p], :k] = L[[p, k], :k]
        L[k + 1:, k] = U[k + 1:, k] / U[k, k]
        U[k + 1:, k:] -= np.outer(L[k + 1:, k], U[k, k:])
    return P, L, np.triu(U)


def forward_substitution(L, b):
    y = np.zeros(len(b))
    for i in range(len(b)):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y


def back_substitution(U, y):
    x = np.zeros(len(y))
    for i in range(len(y) - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def lu_solve(A, b):
    """Solve A x = b via LU. Raises ValueError if A is singular."""
    P, L, U = lu_decompose(A)
    if np.any(np.abs(np.diag(U)) < 1e-12):
        raise ValueError("matrix is singular to working precision")
    return back_substitution(U, forward_substitution(L, P @ np.asarray(b, dtype=float)))


def determinant(A):
    """det(A) from the LU factors; the sign comes from the parity of the row swaps."""
    P, _, U = lu_decompose(A)
    perm, n = np.argmax(P, axis=1), len(P)
    seen, sign = [False] * n, 1
    for i in range(n):
        if not seen[i]:
            j, length = i, 0
            while not seen[j]:
                seen[j] = True
                j = perm[j]
                length += 1
            sign *= -1 if length % 2 == 0 else 1
    return sign * float(np.prod(np.diag(U)))
