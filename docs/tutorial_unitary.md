# Tutorial: Generating Random Unitary Matrices with `random_unitary_matrix`

`random_unitary_matrix` produces a random unitary (complex) matrix \(\mathbf{U}\) of size \(n\times n\). Unitary matrices satisfy \(\mathbf{U}^\dagger \mathbf{U}=\mathbf{I}\) and have determinant of unit modulus. They appear in quantum mechanics, signal processing, and any context where complex inner‑product spaces are involved.

## Basic Usage
```python
from sympy_matrices_random import random_unitary_matrix

# 3×3 unitary matrix (default settings)
U = random_unitary_matrix(3)
print(U)
print("U.H * U =", U.H * U)  # should be identity
print("det(U) =", U.det())
```

## Customizing the Complex Units
The `units` argument defines the set of complex numbers (roots of unity) from which the elementary rotation cells are drawn. By default SymPy’s standard roots of unity are used.
```python
from sympy import I, exp, pi

# Use a custom set of roots of unity
custom_units = (I, exp(I*pi/4), -I, exp(-I*pi/4))
U = random_unitary_matrix(4, units=custom_units)
print(U)
```

## Specifying a Diagonal Normal Form
If you provide a `spec` iterable, the function builds a diagonal matrix \(\mathbf{D}\) whose entries are the supplied complex phases (must have modulus 1). The final unitary matrix is then
\[\mathbf{U}=\mathbf{S}^\dagger \mathbf{D}\,\mathbf{S}\]
where \(\mathbf{S}\) is a random unitary matrix.
```python
from sympy import I, exp, pi

# Define a set of complex phases (unit modulus)
phases = [exp(I*pi/3), exp(-I*pi/3), 1]
U = random_unitary_matrix(3, spec=phases)
print(U)
```

## Number of Rotations (`k`)
`k` controls how many elementary complex rotation cells are multiplied to form the random matrix. The default is `2*dim`.
```python
U = random_unitary_matrix(5, k=30)  # a denser random unitary matrix
```

## Summary of Parameters
| Parameter | Type | Description |
|---|---|---|
| `dim` | `int` | Matrix dimension (n). |
| `spec` | `iterable` | Complex phases (unit modulus) for the diagonal normal form \(\mathbf{D}\). |
| `units` | `iterable` | Set of complex numbers used to build the elementary rotation cells. |
| `k` | `int` | Number of rotation cells; default `2*dim`. |

## Quick Recipes
* **Random 2×2 unitary matrix** (a simple SU(2) element):
  ```python
  U = random_unitary_matrix(2)
  ```
* **Unitary matrix with a prescribed eigenvalue** (using `spec`):
  ```python
  from sympy import exp, I, pi
  eigen = exp(I*pi/4)  # e^{iπ/4}
  U = random_unitary_matrix(3, spec=[eigen, 1, 1])
  ```
* **Force determinant to be +1** (optional for special unitary matrices):
  ```python
  U = random_unitary_matrix(3)
  det = U.det()
  if det != 1:
      # Normalize by dividing by the n‑th root of the determinant
      U = U / det**(1/3)
  ```

These examples illustrate how to generate unitary matrices suitable for quantum‑state simulations, testing algorithms that require complex orthonormal bases, or symbolic investigations of matrix properties.
