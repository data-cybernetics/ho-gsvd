# Higher-Order GSVD & CSD - Python Implementation

This repository provides a **NumPy implementation** of the **Higher-Order Generalized Singular Value Decomposition (HO-GSVD)** and the **Higher-Order Cosine-Sine Decomposition (HO-CSD)**, including support for **rank-deficient matrices**, as presented in:

> **Kempf, Idris; Goulart, Paul J.; Duncan, Stephen R.**  
> *A Higher-Order Generalized Singular Value Decomposition for Rank Deficient Matrices*  
> arXiv:2102.09822 (2021).  
> https://arxiv.org/abs/2102.09822

This project is a **Python translation** of the original MATLAB toolbox available at https://github.com/kmpape/HO-GSVD .

---

## Usage Example

Below is a minimal example that computes the HO-GSVD of three matrices stacked vertically into a single block matrix `A`:

```python
import numpy as np
from hogsvd import hogsvd

# Block sizes
m = np.array([5, 6, 4])
n = 3

# Construct a stacked A = [A1; A2; A3]
rng = np.random.default_rng(0)
A = rng.standard_normal((np.sum(m), n)) + 1j * rng.standard_normal((np.sum(m), n))

# Compute HO-GSVD
res = hogsvd(A, m)

print("U shape:", res.u.shape)
print("S shape:", res.s.shape)
print("V shape:", res.v.shape)
print("Isolated classes:", res.iso_classes)
```

Or compute the HO-CSD of a Q-stacked matrix:

```python
from hogsvd import hocsd

Q = np.linalg.qr(A, mode="reduced")[0]
res_csd = hocsd(Q, m)

print("Right unitary Z:", res_csd.z)
print("Tau eigenvalues:", np.diag(res_csd.tau))
```

---

## Running the Test Suite

Simply run:

```bash
pytest -q
```

Tests include:

* shape & orthogonality checks
* block-wise reconstruction
* full-rank and rank-deficient scenarios
* padding behavior
* error handling

---

## References

Please cite the original paper if you use this code in academic work:

```
@misc{hogsvd,
  doi       = {10.48550/ARXIV.2102.09822},
  url       = {https://arxiv.org/abs/2102.09822},
  author    = {Kempf, Idris and Goulart, Paul J. and Duncan, Stephen R.},
  title     = {A Higher-Order Generalized Singular Value Decomposition for Rank Deficient Matrices},
  publisher = {arXiv},
  year      = {2021},
}
```

---

## Acknowledgment

This project is based on and closely follows the design of the **official MATLAB HO-GSVD implementation**:

> [https://github.com/kmpape/HO-GSVD](https://github.com/kmpape/HO-GSVD)

Many thanks to the authors for releasing the reference code and the paper that inspired this work.

---
