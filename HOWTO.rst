Generating Random Matrices
--------------------------

This tutorial explains how to use the `random_matrix` function to generate 
structured random matrices that are useful for linear algebra experiments, 
testing, and theoretical verification.

Introduction
~~~~~~~~~~~~

The `random_matrix` function doesn't just generate random numbers; 
it creates a square matrix with specific algebraic properties, 
such as a predefined **rank** or a specific **spectrum** (eigenvalues).

It constructs the matrix using the formula:
$$\mathbf{M} = \mathbf{S}^{-1} \mathbf{J} \mathbf{S}$$
where $\mathbf{J}$ is a matrix in Jordan normal form 
and $\mathbf{S}$ is a random invertible transformation matrix.

Basic Usage
~~~~~~~~~~~

Simple Random Matrix
^^^^^^^^^^^^^^^^^^^^

By default, calling `random_matrix(dim)` generates a full-rank random matrix 
with integer entries.

.. code:: python

  from sympy_matrices_random import random_matrix

  # Generate a 3x3 random matrix
  m = random_matrix(3)
  print(m)


Controlling the Rank
^^^^^^^^^^^^^^^^^^^^

You can specify the rank of the resulting matrix. This is particularly useful 
for testing algorithms that handle singular matrices.

.. code:: python

  # Generate a 4x4 matrix with rank 2
  m = random_matrix(4, rank=2)
  print(f"Rank: {m.rank()}")  # Output: 2


Specifying Eigenvalues (The Spectrum)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `spec` argument allows you to define the eigenvalues of the matrix. 
You can use **SymPy symbols** to create matrices with symbolic eigenvalues.

- **Simple list**: `spec=[1, 2, 3]` creates a matrix with eigenvalues choosen randomly from this list.
- **Symbolic Spectrum**: Pass symbols to generate a matrix whose properties depend on those variables.
- **Jordan Blocks**: To create larger Jordan blocks, repeat the eigenvalue or provide a tuple `(block_size, value)`.

.. code:: python

  from sympy import symbols
  from sympy_matrices_random import random_matrix

  a, b, c = symbols('a b c')
  
  # Generate a 3x3 matrix with symbolic eigenvalues a, b, and c
  m = random_matrix(3, spec=[a, b, c])
  print(m)
  print(m.eigenvals())  # has eigenvalues in {a, b, c}


Forcing a Jordan Normal Form
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you set `k=0`, the function skips the transformation matrix $\mathbf{S}$ 
and returns the Jordan matrix $\mathbf{J}$ directly. 
This is excellent for creating symbolic templates of matrices.

.. code:: python

  from sympy import symbols
  from sympy_matrices_random import random_matrix
  
  lambda_ = symbols('lambda')
  
  # Create a symbolic Jordan block of size 3 for eigenvalue lambda
  m = random_matrix(3, spec=[lambda_, lambda_, lambda_], k=0)
  print(m)
  # Result: Matrix([[lambda_, 1, 0], [0, lambda_, 1], [0, 0, lambda_]])


Specifying the rank as well possibly adds additional $0$ eigenvalues.

.. code:: python

  m = random_matrix(4, spec=[lambda_, lambda_, lambda_], rank=3, =0)
  print(m)
  # Result: Matrix([[lambda_, 1, 0, 0], [0, lambda_, 1, 0], [0, 0, lambda_, 0], [0, 0, 0, 0]])


Customizing the Transformation Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The matrices $\mathbf{S}$ and $\mathbf{S}^{-1}$ are built using elementary matrices. 
You can control the values used in these matrices via `scalars` and `units`.

.. code:: python

  from sympy import sqrt, symbols
  
  u = symbols('u')
  
  # Use specific values for the random transformation
  m = random_matrix(3, scalars=(sqrt(2), 2), units=(u, 1/u))
  print(m)
  
  # No specified spectrum but matrix theory tells
  print(m.det())  # must be a power of u


Summary of Parameters
~~~~~~~~~~~~~~~~~~~~~

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `dim` | `int` | Dimension of the square matrix ($n \times n$). |
| `spec` | `list/tuple` | Desired eigenvalues. Use repeats or `(size, val)` for Jordan blocks. |
| `rank` | `int` | Target rank of the matrix. |
| `triangular` | `bool` | If `True`, returns an upper triangular matrix. |
| `k` | `int` | Number of elementary matrices used for $\mathbf{S}$. Set to `0` for $\mathbf{J}$ only. |
| `scalars` | `iterable` | Values used for non-diagonal entries in elementary matrices. |
| `units` | `iterable` | Values used for diagonal entries (must be invertible). |


Common Recipes
~~~~~~~~~~~~~~

**Create a random idempotent matrix ($\mathbf{M}^2 = \mathbf{M}$):**
Set `spec=[1]` and a specific `rank`.

.. code:: python

  m = random_matrix(3, spec=[1], rank=2)
  assert (m*m == m)


**Create a random nilpotent matrix ($\mathbf{M}^n = 0$):**
Set `spec=[0]` and a specific `rank`.

.. code:: python

  from sympy import zeros
  
  m = random_matrix(3, spec=[0], rank=2)
  assert (m**3 == zeros(3))
