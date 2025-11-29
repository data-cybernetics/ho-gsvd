import numpy as np
import numpy.testing as npt
import pytest

from ho_gsvd import hocsd, hogsvd, _split_offsets # pyright: ignore[reportPrivateUsage]


def _make_random_q(m: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Construct a random stacked Q with orthonormal columns:
        Q in C^{sum(m) x n}, Q^* Q = I_n
    and block structure defined by m.
    """
    total_rows = int(np.sum(m))
    # Random tall matrix, then QR
    a = rng.standard_normal((total_rows, n)) + 1j * rng.standard_normal((total_rows, n))
    q, _ = np.linalg.qr(a, mode="reduced")
    return q


def _make_random_a_full_rank(m: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Random stacked A = [A1; ...; AN] with full column rank n.
    """
    total_rows = int(np.sum(m))
    a = rng.standard_normal((total_rows, n)) + 1j * rng.standard_normal((total_rows, n))
    # Ensure full column rank by perturbation
    a += 1e-3 * np.eye(total_rows, n, dtype=a.dtype)
    return a


def _make_random_a_rank_deficient(m: np.ndarray, n: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    """
    Construct A with prescribed rank < n by projecting onto a rank-dimensional subspace.
    """
    if rank >= n:
        raise ValueError("rank must be < n for rank-deficient test.")
    total_rows = int(np.sum(m))

    # Random basis for rank-dimensional subspace in C^n
    basis = rng.standard_normal((n, rank)) + 1j * rng.standard_normal((n, rank))
    basis, _ = np.linalg.qr(basis, mode="reduced")  # n x rank, orthonormal columns

    # Random coefficients for each row in that subspace
    coeffs = rng.standard_normal((total_rows, rank)) + 1j * rng.standard_normal((total_rows, rank))
    a = coeffs @ basis.conj().T  # shape (sum(m), n), rank = rank
    return a


def test_hocsd_shapes_and_orthogonality():
    rng = np.random.default_rng(1234)
    m = np.array([5, 7, 6], dtype=np.int64)
    n = 4

    q = _make_random_q(m, n, rng)
    res = hocsd(q, m, disable_warnings=True)

    # Shape checks
    total_rows = int(np.sum(m))
    num_blocks = len(m)
    assert res.u.shape == (total_rows, n)
    assert res.s.shape == (num_blocks * n, n)
    assert res.z.shape == (n, n)
    assert res.tau.shape == (n, n)

    # Z should be unitary: Z^* Z ≈ I
    ident = res.z.conj().T @ res.z
    npt.assert_allclose(ident, np.eye(n, dtype=res.z.dtype), rtol=1e-10, atol=1e-10)

    # Tau should be diagonal
    npt.assert_allclose(res.tau, np.diag(np.diag(res.tau)), rtol=1e-12, atol=1e-12)


def test_hocsd_reconstruction_per_block():
    rng = np.random.default_rng(5678)
    m = np.array([4, 3], dtype=np.int64)
    n = 3

    q = _make_random_q(m, n, rng)
    res = hocsd(q, m, disable_warnings=True)

    m_arr, offsets = _split_offsets(m)
    num_blocks = len(m_arr)

    # For each block i: Qi ≈ Ui Si Z^*
    for i in range(num_blocks):
        row_slice = slice(offsets[i], offsets[i + 1])
        qi = q[row_slice, :]
        ui = res.u[row_slice, :]
        si = res.s[i * n : (i + 1) * n, :]

        qi_recon = ui @ si @ res.z.conj().T
        npt.assert_allclose(qi_recon, qi, rtol=1e-10, atol=1e-10)


def test_hocsd_s_blocks_are_diagonal():
    rng = np.random.default_rng(1010)
    m = np.array([4, 4], dtype=np.int64)
    n = 3

    q = _make_random_q(m, n, rng)
    res = hocsd(q, m, disable_warnings=True)

    num_blocks = len(m)
    for i in range(num_blocks):
        si = res.s[i * n : (i + 1) * n, :]
        # Each block Si should be diagonal
        npt.assert_allclose(si, np.diag(np.diag(si)), rtol=1e-12, atol=1e-12)


def test_hogsvd_full_rank_reconstruction():
    rng = np.random.default_rng(999)
    m = np.array([5, 6], dtype=np.int64)
    n = 4

    a = _make_random_a_full_rank(m, n, rng)
    res = hogsvd(a, m, disable_warnings=True)

    total_rows = int(np.sum(m))
    assert res.u.shape == (total_rows, n)
    assert res.s.shape == (len(m) * n, n)
    assert res.v.shape == (n, n)

    # V is not expected to be unitary in this construction.
    # Instead, check that V is well-conditioned (full rank).
    v_rank = np.linalg.matrix_rank(res.v)
    assert v_rank == n

    # Reconstruction per block: Ai ≈ Ui Si V^*
    m_arr, offsets = _split_offsets(m)
    for i in range(len(m_arr)):
        row_slice = slice(offsets[i], offsets[i + 1])
        ai = a[row_slice, :]
        ui = res.u[row_slice, :]
        si = res.s[i * n : (i + 1) * n, :]

        ai_recon = ui @ si @ res.v.conj().T
        np.testing.assert_allclose(ai_recon, ai, rtol=1e-10, atol=1e-10)


def test_hogsvd_rank_deficient_padding_and_reconstruction():
    rng = np.random.default_rng(4242)
    m = np.array([4, 5], dtype=np.int64)
    n = 4
    rank = 3  # rank-deficient

    a = _make_random_a_rank_deficient(m, n, rank, rng)
    res = hogsvd(a, m, disable_warnings=True)

    total_rows = int(np.sum(m))
    # After padding, hogsvd removes the padding again, so shapes match original stacked size
    assert res.u.shape == (total_rows, n)
    assert res.s.shape == (len(m) * n, n)
    assert res.v.shape == (n, n)

    # Check reconstruction accuracy (should still be good even though A was rank-deficient)
    m_arr, offsets = _split_offsets(m)
    for i in range(len(m_arr)):
        row_slice = slice(offsets[i], offsets[i + 1])
        ai = a[row_slice, :]
        ui = res.u[row_slice, :]
        si = res.s[i * n : (i + 1) * n, :]

        ai_recon = ui @ si @ res.v.conj().T
        # Slightly looser tolerance because of padding + rank deficiency
        npt.assert_allclose(ai_recon, ai, rtol=1e-9, atol=1e-9)


def test_hocsd_invalid_m_raises():
    rng = np.random.default_rng(2025)
    m = np.array([3, 3], dtype=np.int64)
    n = 3

    # Make Q with sum(m) + 1 rows to trigger mismatch
    q = _make_random_q(m, n, rng)
    q_bad = np.vstack([q, np.zeros((1, n), dtype=q.dtype)])

    with pytest.raises(ValueError):
        _ = hocsd(q_bad, m, disable_warnings=True)


def test_hogsvd_invalid_m_raises():
    rng = np.random.default_rng(2026)
    m = np.array([2, 2], dtype=np.int64)
    n = 3

    a = _make_random_a_full_rank(m, n, rng)
    # Change m so sum(m) is wrong
    m_bad = np.array([3, 2], dtype=np.int64)
    with pytest.raises(ValueError):
        _ = hogsvd(a, m_bad, disable_warnings=True)
