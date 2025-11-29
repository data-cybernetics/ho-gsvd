from dataclasses import dataclass

import warnings
import numpy as np


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class HocsdResult:
    """
    Result of the HOCSD computation.

    Attributes
    ----------
    u :
        Stacked left factors U = [U1; ...; UN], shape (sum(m), n).
    s :
        Block-diagonal generalized singular values, shape (N * n, n).
        Block i (0-based) is s[i*n : (i+1)*n, :].
    z :
        Right unitary matrix, shape (n, n), with z^* z = I.
    tau :
        Diagonal matrix of eigenvalues of T, shape (n, n).
    taumin :
        Theoretical minimum eigenvalue of T.
    taumax :
        Theoretical maximum eigenvalue of T.
    iso_classes :
        1D int64 array of length n_iso (dimension of isolated subspace).
        iso_classes[k] is the index (0-based) of the matrix Qi that has
        a unit generalized singular value associated to the k-th isolated
        direction.
    """
    u: np.ndarray
    s: np.ndarray
    z: np.ndarray
    tau: np.ndarray
    taumin: float
    taumax: float
    iso_classes: np.ndarray


@dataclass
class HogsvdResult:
    """
    Result of the HO-GSVD computation.

    Attributes
    ----------
    u :
        Stacked left factors U = [U1; ...; UN], shape (sum(m), n)
        or (sum(m) - rank_def_A, n) if padding is removed.
    s :
        Block-diagonal generalized singular values, shape (N * n, n),
        or ((N * n - n), n) after removing padding.
    v :
        Shared right factor, shape (n, n).
    tau :
        Diagonal matrix of eigenvalues of T, shape (n, n).
    taumin :
        Theoretical minimum eigenvalue of T.
    taumax :
        Theoretical maximum eigenvalue of T.
    iso_classes :
        Isolated subspace class indices (0-based), same convention as in HocsdResult.
    """
    u: np.ndarray
    s: np.ndarray
    v: np.ndarray
    tau: np.ndarray
    taumin: float
    taumax: float
    iso_classes: np.ndarray


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_offsets(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a length-N sequence m with row-block sizes, return:

    - m_arr: 1D int64 array of block sizes
    - offsets: 1D int64 array of length N+1 such that block i corresponds
      to rows offsets[i]:offsets[i+1].
    """
    m_arr = np.asarray(m, dtype=np.int64)
    if m_arr.ndim != 1:
        raise ValueError("m must be a 1D array-like of block sizes.")
    offsets = np.concatenate(([0], np.cumsum(m_arr)))
    return m_arr, offsets


def _get_block(a: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    """
    View of the `index`-th block in a stacked matrix `a`,
    using precomputed offsets. `index` is 0-based.
    """
    return a[offsets[index] : offsets[index + 1], :]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hocsd(
    q: np.ndarray,
    m: np.ndarray,
    ppi: float = 1e-3,
    zerotol: float = 1e-14,
    eps_rel_iso: float = 1e-6,
    disable_warnings: bool = False,
) -> HocsdResult:
    """
    Higher-Order Cosine-Sine Decomposition for (possibly) rank-deficient case.

    This is a NumPy translation of the MATLAB `hocsd` function.

    Parameters
    ----------
    q :
        Stacked matrix Q = [Q1; ...; QN] of shape (sum(m), n), with each Qi
        having m[i] rows and n columns. The matrix should satisfy approximately
        Q^* Q ≈ I_n.
    m :
        1D sequence of block row sizes, length N. Block i corresponds to
        rows sum(m[:i]) : sum(m[:i+1]) in q.
    ppi :
        Regularization parameter for the matrices Qi'Qi + ppi * I, default 1e-3.
    zerotol :
        Tolerance used to decide zero generalized singular values, default 1e-14.
    eps_rel_iso :
        Relative tolerance for identifying the isolated subspace, default 1e-6.
    disable_warnings :
        If True, suppresses diagnostic warnings.

    Returns
    -------
    HocsdResult
        Dataclass containing (u, s, z, tau, taumin, taumax, iso_classes).

    Complexity (rough, leading terms)
    ---------------------------------
    Let:
        - n   : number of columns
        - N   : number of blocks
        - M   : total number of rows = sum(m)

    Main costs:
        - For each block i, QR of (m[i] + n) x n matrix:
              O( sum_i (m[i] + n) * n^2 )  ≈ O(M n^2 + N n^3)
        - SVD of Rhat (shape n x (N n)):
              O(N n^3)
        - Per-block SVD in zero-GSV handling (worst case):
              O( sum_i m[i] n^2 ) ≈ O(M n^2)

    Overall:
        Time   : O(M n^2 + N n^3)
        Memory : O(M n + N n^2)
    """
    warn_eps_iso = 1e-6
    warn_cond = 1e6

    q = np.asarray(q)
    if q.ndim != 2:
        raise ValueError("q must be a 2D array.")

    m_arr, offsets = _split_offsets(m)
    num_blocks = len(m_arr)
    n = q.shape[1]

    if offsets[-1] < n:
        raise ValueError(
            f"sum(m)={offsets[-1]} < n={n}. Rank(Q) = {n} required."
        )
    if q.shape[0] != offsets[-1]:
        raise ValueError(
            f"size(q, 1)={q.shape[0]} != sum(m)={offsets[-1]}."
        )

    # Prepare list of Qi blocks as views (no copies)
    q_blocks = [_get_block(q, offsets, i) for i in range(num_blocks)]

    # Check Q'*Q ~ I (if warnings enabled)
    if not disable_warnings:
        eye_n = np.eye(n, dtype=q.dtype)
        diff = q.conj().T @ q - eye_n
        err_q = np.linalg.norm(diff) / n
        if err_q >= zerotol:
            warnings.warn(
                f"Expected norm(Q'*Q-eye(n))/n={err_q:e} < ZEROTOL={zerotol:e}.",
                RuntimeWarning,
            )

    # --- Compute eigenvectors/eigenvalues of T via SVD of Rhat ---
    rhat = np.zeros((n, num_blocks * n), dtype=q.dtype)
    sqrt_ppi = np.sqrt(ppi)
    eye_n = np.eye(n, dtype=q.dtype)

    for block_index, q_block in enumerate(q_blocks):
        # Equivalent to [Qi; sqrt_ppi * I_n], reduced QR
        stacked = np.vstack((q_block, sqrt_ppi * eye_n))
        _, rhat_i = np.linalg.qr(stacked, mode="reduced")

        # Use solve instead of explicit inverse
        inv_rhat_i = np.linalg.solve(rhat_i, eye_n)
        rhat[:, block_index * n : (block_index + 1) * n] = inv_rhat_i

        if not disable_warnings:
            cond_rhat_i = np.linalg.cond(rhat_i)
            if cond_rhat_i >= warn_cond:
                warnings.warn(
                    f"For i={block_index}, cond(Rhat_i)={cond_rhat_i:e}",
                    RuntimeWarning,
                )

    # SVD of Rhat: Rhat = Z * diag(singular_values) * Vh
    z, singular_values, _ = np.linalg.svd(rhat, full_matrices=False)
    tau_diag = (singular_values**2) / num_blocks
    tau = np.diag(tau_diag)

    taumin = 1.0 / (1.0 / num_blocks + ppi)
    taumax = (
        (num_blocks - 1) / (num_blocks * ppi)
        + 1.0 / (num_blocks * (1.0 + ppi))
    )

    # Indices of the isolated subspace
    ind_iso = np.abs(taumax - tau_diag) <= (taumax - taumin) * eps_rel_iso

        # Align isolated eigenvectors (Algorithm 6.3)
    iso_classes = np.array([], dtype=np.int64)
    if np.any(ind_iso):
        # Explicitly tell the type checker this is an ndarray
        z_iso: np.ndarray = z[:, ind_iso]  # shape (n, n_iso)
        n_iso = z_iso.shape[1]
        z_iso_new = np.zeros_like(z_iso)
        iso_classes = np.zeros(n_iso, dtype=np.int64)

        z_iter: np.ndarray = z_iso
        # Sort classes according to largest gain in the i-th subspace
        for i in range(n_iso - 1):
            all_s = np.empty(num_blocks, dtype=float)
            for block_index, q_block in enumerate(q_blocks):
                # Help PyLance infer the type of the argument to norm
                prod: np.ndarray = q_block @ z_iter
                all_s[block_index] = float(np.linalg.norm(prod, ord=2))

            ind_sorted = np.argsort(all_s)[::-1]
            iso_classes[i] = ind_sorted[0]

            # Cast iso_classes[i] to a plain int so list indexing is well-typed
            iso_idx = int(iso_classes[i])
            q_iso: np.ndarray = q_blocks[iso_idx]
            # SVD of q_iso @ z_iter; need right singular vectors
            prod_iso: np.ndarray = q_iso @ z_iter
            _, _, x_iso_h = np.linalg.svd(prod_iso, full_matrices=False)
            x_iso: np.ndarray = x_iso_h.conj().T

            z_iso_new[:, i] = z_iter @ x_iso[:, 0]
            z_iter = z_iter @ x_iso[:, 1:]  # reduce dimension by one

        # Last vector (z_iter is now a single vector)
        all_s = np.empty(num_blocks, dtype=float)
        for block_index, q_block in enumerate(q_blocks):
            prod_last: np.ndarray = q_block @ z_iter
            all_s[block_index] = float(np.linalg.norm(prod_last, ord=2))

        ind_sorted = np.argsort(all_s)[::-1]
        iso_classes[-1] = ind_sorted[0]
        z_iso_new[:, -1] = z_iter.ravel()


        # Replace isolated subspace in z
        z[:, ind_iso] = z_iso_new

        if not disable_warnings:
            err_z = np.linalg.norm(
                np.eye(n, dtype=z.dtype) - z.conj().T @ z, ord=2
            )
            if err_z > warn_eps_iso:
                warnings.warn(
                    f"Rotated Z is not orthogonal, "
                    f"norm(I - Z^T Z)={err_z:e}.",
                    RuntimeWarning,
                )

    # Check orthogonality of z
    z_not_orthogonal = (
        np.linalg.norm(
            np.eye(n, dtype=z.dtype) - z.conj().T @ z, ord=2
        )
        > warn_eps_iso
    )

    # --- Compute U and S blocks ---
    s = np.zeros((num_blocks * n, n), dtype=q.dtype)
    u = np.zeros((offsets[-1], n), dtype=q.dtype)

    for block_index, q_block in enumerate(q_blocks):
        row_slice = slice(offsets[block_index], offsets[block_index + 1])

        if z_not_orthogonal:
            # Solve Z^* B^T = Q^T  => B = (Z^*)^{-1} Q
            b = np.linalg.solve(z.conj().T, q_block.T).T
        else:
            b = q_block @ z

        # Column 2-norms
        s_vec = np.linalg.norm(b, axis=0)
        ind_pos = s_vec > zerotol

        # Non-zero generalized singular values
        if np.any(ind_pos):
            scale = 1.0 / s_vec[ind_pos]  # shape (npos,)
            # Correct in-place write (avoid writing through a temporary view)
            u[row_slice, ind_pos] = b[:, ind_pos] * scale

        zero_mask = ~ind_pos
        num_zero = int(np.count_nonzero(zero_mask))

        if num_zero > 0:
            # Full SVD to mimic MATLAB behavior
            u_qi, svals, _ = np.linalg.svd(q_block, full_matrices=True)

            # Build diag(SQi) of length m_i to detect zero singular values
            diag_s_qi = np.zeros(u_qi.shape[0], dtype=svals.dtype)
            diag_s_qi[: len(svals)] = svals
            ind_zero_i = diag_s_qi <= zerotol
            num_zero_i = int(np.count_nonzero(ind_zero_i))

            if num_zero_i == 0:
                # Substitute normalized columns of q_block for zero GSVs
                q_tmp = q_block[:, zero_mask]
                norms = np.linalg.norm(q_tmp, axis=0)
                norms[norms <= zerotol] = 1.0
                u[row_slice, zero_mask] = q_tmp / norms
            else:
                # Substitute left singular vectors associated with zero SVs
                u_i2 = u_qi[:, ind_zero_i]
                if num_zero_i < num_zero:
                    reps = int(np.ceil(num_zero / num_zero_i))
                    u_i2 = np.tile(u_i2, (1, reps))
                u[row_slice, zero_mask] = u_i2[:, :num_zero]

        # Store diag(s_vec) in block rows
        s[block_index * n : (block_index + 1) * n, :] = np.diag(s_vec)

    if not disable_warnings:
        for block_index in range(num_blocks):
            u_block = u[offsets[block_index] : offsets[block_index + 1], :]
            s_block = s[block_index * n : (block_index + 1) * n, :]
            q_block = q_blocks[block_index]
            reconstruction = u_block @ s_block @ z.conj().T
            err_block = float(np.sum(np.abs(reconstruction - q_block)))
            if err_block >= 1e-12:
                warnings.warn(
                    f"HOCSD: reconstruction error={err_block:e} "
                    f"for matrix {block_index}",
                    RuntimeWarning,
                )

    return HocsdResult(
        u=u,
        s=s,
        z=z,
        tau=tau,
        taumin=taumin,
        taumax=taumax,
        iso_classes=iso_classes,
    )


def hogsvd(
    a: np.ndarray,
    m: np.ndarray,
    rank_tol_a: float = 1e-14,
    ppi: float = 1e-3,
    zerotol: float = 1e-14,
    eps_rel_iso: float = 1e-6,
    disable_warnings: bool = False,
) -> HogsvdResult:
    """
    Higher-Order Generalized SVD (HO-GSVD) of stacked matrices.

    This is a NumPy translation of the MATLAB `hogsvd` function.
    It wraps `hocsd` and adds rank-deficiency padding for A.

    Parameters
    ----------
    a :
        Stacked matrix A = [A1; ...; AN] of shape (sum(m), n), with each Ai
        having m[i] rows and n columns.
    m :
        1D sequence of block row sizes, length N. Block i corresponds to
        rows sum(m[:i]) : sum(m[:i+1]) in a.
    rank_tol_a :
        Rank tolerance for deciding rank-deficiency of A, default 1e-14.
    ppi, zerotol, eps_rel_iso, disable_warnings :
        Passed through to hocsd.

    Returns
    -------
    HogsvdResult
        Dataclass containing (u, s, v, tau, taumin, taumax, iso_classes).

    Complexity (rough, leading terms)
    ---------------------------------
    Let:
        - n   : number of columns
        - N   : number of blocks
        - M   : total number of rows = sum(m)

    Main costs:
        - SVD of A (M x n) for rank estimation: O(M n^2)
        - QR of padded A: O(M n^2)
        - Call to hocsd(Q, m_padded):
              O(M n^2 + N n^3)

    Overall:
        Time   : O(M n^2 + N n^3)
        Memory : O(M n + N n^2)
    """
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError("a must be a 2D array.")

    m_arr, offsets = _split_offsets(m)
    n = a.shape[1]

    if a.shape[0] != offsets[-1]:
        raise ValueError(
            f"size(a, 1)={a.shape[0]} != sum(m)={offsets[-1]}."
        )

    # Rank deficiency of A
    singular_values_a = np.linalg.svd(a, compute_uv=False)
    rank_def_a = int(n - np.sum(singular_values_a > rank_tol_a))

    if rank_def_a == 0:
        a_padded = a
        m_padded = m_arr
    else:
        if not disable_warnings:
            warnings.warn(
                f"Provided rank-deficient A with rank(A)={n - rank_def_a} < n={n}. "
                "Padding A.",
                RuntimeWarning,
            )
        # Full SVD to get right singular vectors
        _, _, v_a_h = np.linalg.svd(a, full_matrices=True)
        v_a = v_a_h.conj().T  # n x n
        pad_rows = v_a[:, -rank_def_a:].T  # rank_def_a x n
        a_padded = np.vstack((a, pad_rows))
        m_padded = np.concatenate((m_arr, [rank_def_a]))
        _, offsets = _split_offsets(m_padded)

    # QR decomposition of A_padded
    q, r = np.linalg.qr(a_padded, mode="reduced")

    # Call hocsd with explicit parameters
    hocsd_result = hocsd(
        q,
        m_padded,
        ppi=ppi,
        zerotol=zerotol,
        eps_rel_iso=eps_rel_iso,
        disable_warnings=disable_warnings,
    )

    u = hocsd_result.u
    s = hocsd_result.s
    tau = hocsd_result.tau
    taumin = hocsd_result.taumin
    taumax = hocsd_result.taumax
    iso_classes = hocsd_result.iso_classes
    z = hocsd_result.z

    # V = R' * Z
    v = r.conj().T @ z

    # Remove padding, if any
    if rank_def_a > 0:
        u = u[:-rank_def_a, :]
        s = s[:-n, :]  # drop last block of size n

    return HogsvdResult(
        u=u,
        s=s,
        v=v,
        tau=tau,
        taumin=taumin,
        taumax=taumax,
        iso_classes=iso_classes,
    )
