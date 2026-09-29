import numpy as np
import pytest
from linalg import (lu_decompose, lu_solve, determinant, qr_decompose, least_squares,
                    power_iteration, eigh_qr, svd, low_rank)

rng = np.random.default_rng(42)


def test_lu_reconstructs():
    A = rng.standard_normal((6, 6))
    P, L, U = lu_decompose(A)
    assert np.allclose(P @ A, L @ U)
    assert np.allclose(L, np.tril(L)) and np.allclose(U, np.triu(U))


def test_lu_solve_and_det():
    A, b = rng.standard_normal((5, 5)), rng.standard_normal(5)
    assert np.allclose(lu_solve(A, b), np.linalg.solve(A, b))
    assert np.isclose(determinant(A), np.linalg.det(A))


def test_lu_singular():
    with pytest.raises(ValueError):
        lu_solve(np.array([[1.0, 2.0], [2.0, 4.0]]), np.array([1.0, 2.0]))


@pytest.mark.parametrize("shape", [(5, 5), (8, 4), (4, 6)])
def test_qr(shape):
    A = rng.standard_normal(shape)
    Q, R = qr_decompose(A)
    assert np.allclose(Q @ R, A)
    assert np.allclose(Q.T @ Q, np.eye(shape[0]))
    assert np.allclose(R, np.triu(R))


def test_least_squares():
    A, b = rng.standard_normal((20, 3)), rng.standard_normal(20)
    assert np.allclose(least_squares(A, b), np.linalg.lstsq(A, b, rcond=None)[0])


def test_power_iteration():
    M = rng.standard_normal((6, 6))
    A = M @ M.T
    lam, v = power_iteration(A)
    assert np.isclose(lam, np.linalg.eigvalsh(A)[-1])
    assert np.allclose(A @ v, lam * v, atol=1e-6)


def test_eigh_qr():
    M = rng.standard_normal((7, 7))
    A = M + M.T
    vals, vecs = eigh_qr(A)
    assert np.allclose(vals, np.sort(np.linalg.eigvalsh(A))[::-1])
    assert np.allclose(A @ vecs, vecs * vals, atol=1e-8)
    assert np.allclose(vecs.T @ vecs, np.eye(7), atol=1e-8)


@pytest.mark.parametrize("shape", [(6, 6), (9, 4), (4, 9)])
def test_svd(shape):
    A = rng.standard_normal(shape)
    U, s, Vt = svd(A)
    assert np.allclose(U @ np.diag(s) @ Vt, A)
    assert np.allclose(s, np.linalg.svd(A, compute_uv=False))


def test_low_rank_error_matches_theory():
    A = rng.standard_normal((10, 8))
    s = np.linalg.svd(A, compute_uv=False)
    assert np.isclose(np.linalg.norm(A - low_rank(A, 3), 2), s[3])
