# Tutorial: Generating Random Orthogonal Matrices with `random_orthogonal_matrix`

`random_orthogonal_matrix` creates a random orthogonal (real) matrix \(\mathbf{Q}\) of size \(n\times n\). Orthogonal matrices satisfy \(\mathbf{Q}^\top \mathbf{Q}=\mathbf{I}\) and have determinant \(\pm 1\). They are useful for rotations, QR factorizations, and preserving Euclidean norms.

## Basic Usage
```python
from sympy_matrices_random import random_orthogonal_matrix

# 3×3 orthogonal matrix (default settings)
Q = random_orthogonal_matrix(3)
print(Q)
print("Q.T * Q =", Q.T * Q)  # should be identity
print("det(Q) =", Q.det())
```

## Controlling the Rotation Angles
The `angles` argument lets you choose the set of angles from which the elementary Givens rotations are sampled.
```python
from sympy import pi

angles = (pi/6, pi/4, pi/3)  # 30°, 45°, 60°
Q = random_orthogonal_matrix(4, angles=angles)
print(Q)
```

## Specifying a Normal Form
If you provide a `spec` (a list of angles), the function builds a block‑diagonal normal form \(\mathbf{D}\) made of 2×2 rotation blocks and possibly a 1×1 block for odd dimensions. The final matrix is
\[\mathbf{Q}=\mathbf{S}^\top \mathbf{D}\,\mathbf{S}\]
where \(\mathbf{S}\) is a random orthogonal matrix.
```python
from sympy import pi

# Force a specific set of rotation angles in the normal form
spec = (pi/4, pi/2)  # two rotation blocks for a 4×4 matrix
Q = random_orthogonal_matrix(4, spec=spec)
print(Q)
```

## Number of Rotations (`k`)
`k` controls how many elementary Givens rotations are multiplied to form the random orthogonal matrix. Larger `k` yields a matrix that is “more random”. The default is `2*dim`.
```python
Q = random_orthogonal_matrix(5, k=20)  # more rotations than the default 10
```

## Summary of Parameters
| Parameter | Type | Description |
|---|---|---|
| `dim` | `int` | Matrix dimension (n) |
| `spec` | `iterable` | Angles used to build the normal form \(\mathbf{D}\). |
| `angles` | `iterable` | Set of angles to sample from for the random rotations. |
| `k` | `int` | Number of Givens rotations; default `2*dim`. |

## Quick Recipes
* **Random rotation matrix** (2×2):
  ```python
  R = random_orthogonal_matrix(2)
  ```
* **Orthogonal matrix with determinant +1** (force sign):
  ```python
  Q = random_orthogonal_matrix(3)
  if Q.det() == -1:
      Q[0, :] = -Q[0, :]  # flip a row to change sign
  ```
* **Orthogonal matrix with a prescribed eigen‑angle** (using `spec`):
  ```python
  from sympy import pi
  Q = random_orthogonal_matrix(4, spec=(pi/3,))
  ```

These examples show how to generate orthogonal matrices suited for numerical experiments, symbolic derivations, or test‑suite generation.
