"""
sympy.matrices.random
=====================

Utility functions to generate random matrices with prescribed algebraic
properties (rank, spectrum, orthogonality, unitarity).  The core public
functions are :func:`random_matrix`, :func:`random_orthogonal_matrix`,
and :func:`random_unitary_matrix`.
"""

__all__ = [
    "random_matrix",
    "random_orthogonal_matrix",
    "random_unitary_matrix",
]

from sympy import I, Matrix, Mul, conjugate, im, pi, re, rot_givens
from sympy.core.random import sample

_MAX_ITER_JORDAN = 1_000
"""max attempts to find Jordan of requested rank"""

_ELEMENTARY_SCALARS = -1, 1
"""default scalar values for elementary matrices"""

_ELEMENTARY_UNITS = -1, 1
"""default unit values for elementary matrices"""

_ROTATION_ANGLES = 0, pi / 4, pi / 2, pi * 3 / 4
"""default angular values for rotations"""

_ROTATION_UNITS = 1, I, -1 , -I
"""default roots of unity for complex rotations"""


# === random number generator functions ===


def _sample(scalars, k):
    """
    Sample a list of k items.

    Parameters
    ----------
    scalars : iterable
        The set of items to sample from.
    k : int
        Number of items to sample.

    Returns
    -------
    list
        A list of k samples.
    """
    if hasattr(scalars, 'sample'):
        return scalars.sample(k)
    return sample(scalars, k)


def _ssample(scalars):
    """
    Sample a single item.

    Parameters
    ----------
    scalars : iterable
        The set of items to sample from.

    Returns
    -------
    Any
        Sampled item
    """
    return _sample(scalars, k=1)[0]


# === matrices conjugate by products of elementary matrices ===


def _triu(dim, *, scalars=None, units=None, k=None):
    """
    Generate random upper triangular matrix as product of elementary matrices.

    Parameters
    ----------
    dim : int
        Dimension of the matrix.
    scalars : iterable, optional
        Scalar values for elementary matrices.
    units : iterable, optional
        Unit values for elementary matrices.
    k : int, optional
        Number of elementary matrices to multiply.

    Returns
    -------
    sympy.Matrix
        An upper triangular matrix.
    """
    if scalars is None:
        scalars = _ELEMENTARY_SCALARS
    if units is None:
        units = _ELEMENTARY_UNITS
    if k is None:
        k = 2 * dim

    items = [Matrix.eye(dim)]
    for _ in range(k):
        i = _ssample(range(dim))
        j = _ssample(range(i, dim))
        item = Matrix.eye(dim)
        item[i, j] = _ssample(units) if i == j else _ssample(scalars)
        items.append(item)
    return Matrix(Mul(*_sample(items, len(items))))


def _jspec(spec):
    """
    Format a simple list spectrum specification into a tuple of pairs.

    Parameters
    ----------
    spec : iterable
        Spectrum specification.

    Returns
    -------
    list of tuple
        Formatted spectrum as (size, value) pairs.
    """
    # 1, 1, None, 1, 2, (3, 2) --> (2, 1), (1, 1), (1, 2), (3, 2)
    r, s = [], 0
    spec = list(spec)
    v = ()
    for i, v in enumerate(spec):
        if v is None:
            r.append(spec[s: i])
            s = i + 1
            continue
        if not v == spec[s]:
            r.append(spec[s: i])
            s = i
        if isinstance(v, (list, tuple)):
            r.append(tuple(v))
            s = i + 1
    if not isinstance(v, (list, tuple)):
        r.append(spec[s:])
    return [tuple(v) if isinstance(v, tuple) else (len(v), v[0])
            for v in r if v]


def _jordan(dim, *, spec=None, rank=None):
    r"""
    Generate a random upper triangular n x n matrix in Jordan normal form.

    Parameters
    ----------
    dim : int
        Dimension of the matrix.
    spec : iterable, optional
        Spectrum specification. Defaults to $\{1\}$.
    rank : int, optional
        Rank of the matrix. Default is full rank.

    Returns
    -------
    sympy.Matrix
        A matrix in Jordan normal form.

    Raises
    ------
    RuntimeError
        If a Jordan form of the requested rank cannot be found.
    """
    if rank is None:
        rank = dim
    if spec is None:
        p = [1] * rank + [0] * (dim - rank)
        return Matrix.diag(*_sample(p, dim))

    spec = _jspec(spec)

    # split eigenvalues in zero and non zero to ensure proper rank
    zero_spec = [(i, v) for i, v in spec if not v] or [(1, 0)]
    spec = [(i, v) for i, v in spec if v]
    if rank == dim:
        if not spec:
            raise ValueError("Unable to build Jordan matrix of full rank "
                            "with all eigenvalues to be 0.")

    # draw jordan blocks
    # eigenvals 0 to meet dim - rank
    blocks = [_ssample(zero_spec) for _ in range(dim - rank)]
    if spec:
        while sum(i for i, v in blocks) < dim:
            blocks.append(_ssample(spec))

    # adj dimension and rank without changing dim - rank
    while not dim == sum(i for i, v in blocks):
        i, v = blocks.pop(0)  # flush first
        j = max(1, i - 1) if dim < sum(i for i, v in blocks) + i else i + 1
        blocks.append((j, v))  # append

    blocks = _sample(blocks, len(blocks))  # shuffle blocks
    return Matrix.diag(*(Matrix.jordan_block(i, v) for i, v in blocks))


def random_matrix(dim, *, spec=None, scalars=None, units=None,
                  triangular=False, rank=None, k=None):
    r"""
    Generate a random square matrix n x n.

    Such matrix $\mathbf{S}$ may be of a given **rank**
    and may be an upper **triangular** matrix.

    It is constructed as a product of **k** random
    `elementary matrices <https://en.wikipedia.org/wiki/Elementary_matrix>`_
    with with **scalars** arguments for defining row-addtion operations and
    **units** arguments for row-multiplcation operations.

    Eigenvalues may provided as **spec** argument.
    In order to specify not only eigenvalues but a random
    `Jordan matrix <https://en.wikipedia.org/wiki/Jordan_matrix>`_ $\mathbf{J}$
    provide a list of (size, value) pairs to define Jordan blocks.

    Then the matrix is constructed as a product
    $\mathbf{S^{-1} \cdot J \cdot S}$ of an invertible matrix $\mathbf{S}$
    as the product elementary matrices and a Jordan matrix $\mathbf{J}$.

    If **spec** is None (default) only $\mathbf{S}$ will be returned.
    If **k** is $0$ only $\mathbf{J}$ will be returned.

    Parameters
    ----------

    dim : int
        Dimension of the matrix.
    spec : iterable, optional
        Set of values for eigenvalues. Can be a list of eigenvalues or
        tuples of (size, value) for Jordan blocks. Default is None.
    scalars : iterable, optional
        Values used to build the transformation matrix $\mathbf{S}$.
        Defaults to $\{1, -1\}$.
    units : iterable, optional
        Unit values used to build the transformation matrix $\mathbf{S}$.
        Defaults to $\{1, -1\}$.
    triangular : bool, optional
        If True, an upper triangular matrix is returned. Default is False.
    rank : int, optional
        Rank of the matrix. Defaults to full rank.
    k : int, optional
        Number of elementary matrices used to build $\mathbf{S}$.
        Defaults to 2 * dim

    Returns
    -------
    sympy.Matrix
        The generated random matrix.

    See Also
    --------
    jordan_block: SymPy function to create a Jordan block.

    Examples
    --------

    .. ..testsetup::

       >>> from sympy.core.random import rng, seed
       >>> _rng_state = rng.getstate()
       >>> seed(1)

    >>> from sympy_matrices_random import random_matrix

    Generate a 3x3 random matrix

    >>> random_matrix(3)
    Matrix([
    [1,  1,  1],
    [0,  0, -1],
    [0, -1, -1]])

    The argument **k** gives some control of complexity

    >>> m = random_matrix(3, k=42)
    >>> m
    Matrix([
    [0, -4,  1],
    [1, -8,  3],
    [0,  3, -1]])

    By default the result has full rank and integer values.
    And so has the inverse, too.

    >>> m.inv()
    Matrix([
    [ 1, 1,  4],
    [-1, 0, -1],
    [-3, 0, -4]])

    One can specify the rank of the resulting matrix.
    This is particularly useful for testing algorithms
    that handle singular matrices.

    >>> m = random_matrix(4, rank=2)
    >>> m.rank()
    2

    The **spec** argument allows to define the eigenvalues of the matrix.
    As simple list of eigenvalues will define $\mathbf{J}$ as a diagonal
    matrix with random values drawn from this list.

    To promote larger Jordan blocks repeat the eigenvalue or
    specify the size of these blocks for each eigenvalue as
    tuples **(size, value)**.

    >>> m = random_matrix(4, spec=(1, 1, 3))
    >>> m
    Matrix([
    [-1,  2,  0, 0],
    [-4,  5,  0, 0],
    [-1,  1,  2, 1],
    [ 3, -3, -1, 0]])

    >>> m.eigenvals(multiple=True)
    [1, 3, 1, 1]

    >>> m.jordan_form(calc_transform=False)
    Matrix([
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 3]])

    If you set `k=0`, the function skips the transformation matrix $\mathbf{S}$
    and returns the Jordan matrix $\mathbf{J}$ directly.
    This is excellent for creating symbolic templates of matrices.

    You can use **SymPy symbols** to create matrices with symbolic entries.

    >>> from sympy import symbols
    >>> a, b, u, v = symbols('a b u v')

    >>> random_matrix(4, spec=[(2, u), (2, v)], k=0)
    Matrix([
    [u, 1, 0, 0],
    [0, u, 0, 0],
    [0, 0, v, 1],
    [0, 0, 0, v]])

    In general matrix entries of $\mathbf{S}$ are sums and products of
    **scalars** and **units**.

    >>> m = random_matrix(3, scalars=(a, b), units=(u,))
    >>> m
    Matrix([
    [             u, 0,      0],
    [a*b*u**3 + a*u, 1, b*u**2],
    [        a*u**3, 0,   u**2]])

    In contrast, the inverse $\mathbf{S^{-1}}$ consists of
    sums and products of scalars and units as well as fractions of units.

    >>> m.inv()
    Matrix([
    [1/u, 0,       0],
    [ -a, 1,      -b],
    [ -a, 0, u**(-2)]])

    .. ..testcleanup::

       >>> assert not rng.getstate() == _rng_state
       >>> rng.setstate(_rng_state)
       >>> assert rng.getstate() == _rng_state

    """
    if k is None:
        k = 2 * dim
    half = k // 2
    s = _triu(dim, scalars=scalars, units=units, k=k - half)
    t = _triu(dim, scalars=scalars, units=units, k=half)

    if spec is None:
        if rank is not None and rank < dim:
            p = [1] * rank + [0] * (dim - rank)
            t *= Matrix.diag(*_sample(p, dim))
        s *= t if triangular else t.T
        return s

    # remove ev different to 1 from s and t (by construction units)
    s *= Matrix.diag(*(1 / u for u in s.diagonal()))
    t *= Matrix.diag(*(1 / u for u in t.diagonal()))
    s *= t if triangular else t.T
    n = _jordan(dim, spec=spec, rank=rank)
    return s.inv() * n * s


# === matrices conjugate by isometries as product of rotation matrices ===


def _orthogonal(dim, *, angles=None, k=None):
    r"""
    Generate a random orthogonal matrix as product of 2 x 2 rotations cells.

    Constructed as product of random Given rotation matrices with 2 x 2 cells

    .. math::

        \left(\begin{array}{cc}c & -s \\ c & s \end{array}\right)
        \in SO(2).

    Those pairs $c=\cos(\theta)$ and $s=\sin(\theta)$ are are obtained by
    drawing $\theta$ form **angles**.

    Parameters
    ----------
    dim : int
        Dimension of the matrix.
    angles : iterable, optional
        Rotation angles to sample from given as real numbers where $2\pi$
        gives a full circle rotation
    k : int, optional
        Number of rotation matrices to multiply.

    Returns
    -------
    sympy.Matrix
        A random orthogonal matrix.

    See Also
    --------
    rot_givens: Givens rotation matrix.
    random_orthogonal_matrix: Generate a random orthogonal matrix n x n.

    """
    if angles is None:
        angles = _ROTATION_ANGLES
    if k == 0 or dim == 1:
        return Matrix.eye(dim)
    if k is None:
        k = 2 * dim
    ij = [sample(range(dim), 2) for _ in range(k)]
    items = [rot_givens(i, j, _ssample(angles), dim=dim) for i, j in ij]
    return Matrix(Mul(*items))


def random_orthogonal_matrix(dim, *, spec=None, angles=None, k=None):
    r"""
    Generate a random orthogonal matrix n x n.

    An orthogonal matrix $\mathbf{O}$ is a real matrix
    such that $\mathbf{O}^{t} = \mathbf{O}^{-1}$.
    It describes an isometry in n dimensional Euclidian space.

    Build as a product of $2 \times 2$ rotations, known as Givens rotations,
    with rotation angles values drawn randomly from **angles**.
    The number of these rotations is given by **k**.

    If **spec** is given, it should contain a list of real valued angles
    to be drawn from. These define the the isometry normal form
    as a diagonal block matrix $\mathbf{D}$ with $2 \times 2$ rotation blocks.

    Then the resulting isometry matrix will be the conjugate product
    $\mathbf{O}^{-1} \cdot \mathbf{D} \cdot \mathbf{O}$.

    If **spec** is None (default) only $\mathbf{O}$ will be returned.
    If **k** is $0$ only $\mathbf{D}$ will be returned.

    Parameters
    ----------
    dim : int
        Dimension of the matrix.
    spec : iterable, optional
        Set of angles to build the isometry normal form. Default is None.
    angles : iterable, optional
        Rotation angles to sample from. Defaults to fractions $\{\pi n/2 \mid n=1 \dots 4 \}$.
    k : int, optional
        Number of rotations to build the matrix. Defaults to 2 * dim.

    Returns
    -------
    sympy.Matrix
        The generated random orthogonal matrix.

    See Also
    --------
    random_unitary_matrix : Complex generalization of orthogonal matrices.

    Examples
    --------

    .. ..testsetup::

       >>> from sympy.core.random import rng, seed
       >>> _rng_state = rng.getstate()
       >>> seed(0)

    >>> from sympy import expand, pi, sin, symbols
    >>> from sympy.abc import phi, psi
    >>> from sympy_matrices_random import random_orthogonal_matrix

    Generate a 3x3 random orthogonal matrix.

    >>> q = random_orthogonal_matrix(3)
    >>> q
    Matrix([
    [             1/2, sqrt(2)/2,              1/2],
    [-1/2 + sqrt(2)/4,      -1/2,  sqrt(2)/4 + 1/2],
    [ sqrt(2)/4 + 1/2,      -1/2, -1/2 + sqrt(2)/4]])

    >>> expand(q.T * q)
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])

    **Controlling the Rotations.**
    To control how many elementary Givens rotations are multiplied
    to form the random orthogonal matrix, use the **k** argument.
    Larger k yields a matrix that is “more random”. The default is 2*dim.

    >>> Q = random_orthogonal_matrix(3, k=20)  # more rotations than the default 8
    >>> expand(Q)
    Matrix([
    [sqrt(2)/2,          0, sqrt(2)/2],
    [     -1/2,  sqrt(2)/2,       1/2],
    [     -1/2, -sqrt(2)/2,       1/2]])

    The **angles** argument serves as the set of angles
    from which the elementary Givens rotations are sampled.

    >>> angles = pi/3, pi*2/3  # 60°, 120°
    >>> random_orthogonal_matrix(3, angles=angles, k=1)
    Matrix([
    [       1/2, 0, sqrt(3)/2],
    [         0, 1,         0],
    [-sqrt(3)/2, 0,       1/2]])

    Even as symbols

    >>> random_orthogonal_matrix(3, angles=(phi, psi), k=2)
    Matrix([
    [ cos(psi), sin(phi)*sin(psi), sin(psi)*cos(phi)],
    [        0,          cos(phi),         -sin(phi)],
    [-sin(psi), sin(phi)*cos(psi), cos(phi)*cos(psi)]])

    The function takes a list of rotation angles as **spec** argument.
    It constructs a block-diagonal matrix from random $2\times2$
    rotation blocks (plus an optional $1\times1$ block for odd dimensions)
    defining the rotation normal form.

    Such specific rotation angles can be floating point values

    >>> Q = random_orthogonal_matrix(4, spec=angles, k=0)
    >>> expand(Q)
    Matrix([
    [       1/2, sqrt(3)/2,          0,         0],
    [-sqrt(3)/2,       1/2,          0,         0],
    [         0,         0,       -1/2, sqrt(3)/2],
    [         0,         0, -sqrt(3)/2,      -1/2]])

    or symbols

    >>> random_orthogonal_matrix(4, spec=(phi, psi), k=0)
    Matrix([
    [ cos(psi), sin(psi),         0,        0],
    [-sin(psi), cos(psi),         0,        0],
    [        0,        0,  cos(psi), sin(psi)],
    [        0,        0, -sin(psi), cos(psi)]])

    By default the orthogonal matrix is a rotation, i.e. has determinant 1.
    To turn it into a refelction with determinant -1:

    >>> Q = random_orthogonal_matrix(3)
    >>> Q[0, :] = -Q[0, :]  # flip a row to change sign
    >>> Q.det()
    -1

    These examples show how to generate orthogonal matrices suited
    for numerical experiments, symbolic derivations, or test‑suite generation.

    .. ..testcleanup::

       >>> assert not rng.getstate() == _rng_state
       >>> rng.setstate(_rng_state)
       >>> assert rng.getstate() == _rng_state

    """
    s = _orthogonal(dim, angles=angles, k=k)
    if spec is None:
        return s
    blocks = [rot_givens(0, 1, _ssample(spec), dim=2)  # 2 x 2 rotation matrices
              for _ in range(dim // 2)]
    blocks.append([1] * (dim - len(blocks) * 2))
    normal_form = Matrix.diag(*blocks)
    return s.H * normal_form * s


# === matrices conjugate by isometries as product of unitary matrices ===


def _unitary(dim, *, units=None, k=None):
    r"""
    Generate a random unitary matrix.

    Constructed as product of random complex matrices with 2 x 2 cells

    .. math::

        \left(\begin{array}{cc}a & \bar{b} \\ -b & \bar{a} \end{array}\right)
        \in SU(2).

    Those pairs $a$ and $b$ are constructed from three complex roots of unity
    $z, w, u$ by $a=z*\operatorname{Re}(u)$ and $b=w*\operatorname{Im}(u)$.

    Parameters
    ----------
    dim : int
        Dimension of the matrix.
    units : iterable, optional
        Complex units to sample from.
    k : int, optional
        Number of complex rotations to multiply.

    Returns
    -------
    sympy.Matrix
        A random unitary matrix.

    See Also
    --------
    random_unitary_matrix : Complex generalization of orthogonal matrices.

    """
    if units is None:
        units = _ROTATION_UNITS
    if k == 0:
        return Matrix.eye(dim)
    if k is None:
        k = 2 * dim
    if dim == 1:
        items = [_ssample(units) for _ in range(k)]  # 1 x 1 complex rot.
        return Matrix([[Mul(*items)]])
    items = []
    for _ in range(k):
        # draw triple of complex units $(v, w, z)$
        # such that by $c=v*\re(z)$ and $s=w*\im(z)$
        # this yields a complex 2 x 2 rotation matrix, i.e. SU(2) element
        v, w, z = _ssample(units), _ssample(units), _ssample(units)
        c, s = v * re(z), w * im(z)
        i, j = _sample(range(dim), 2)
        item = Matrix.eye(dim)  # gives n x n rotation matrix
        item[i, i], item[i, j] = c, conjugate(s)
        item[j, i], item[j, j] = -s, conjugate(c)
        items.append(item)
    return Matrix(Mul(*items))


def random_unitary_matrix(dim, *, spec=None, units=None, k=None):
    r"""
    Generate a random unitary matrix n x n.

    A unitary matrix $\mathbf{U}$ is complex matrix such that
    $\mathbf{U}^H = \mathbf{U}^{-1}.$
    It describes an isometry in n-dimensional unitary (complex) vectorspace.

    If **spec** is given, it defines the set of random entries of an isometry normal form
    as a diagonal matrix $\mathbf{D}$ with a diagonal of roots of unity.

    Then the resulting isometry matrix will be the conjugate product
    $\mathbf{U}^{-1} \cdot \mathbf{D} \cdot \mathbf{U}$.

    If **spec** is None (default) only $\mathbf{U}$ will be returned.
    If **k** is $0$ only $\mathbf{D}$ will be returned.

    Note, **spec** (the set of eigenvalues) and **units** must
    consist of complex roots of unity only,
    i.e. complex numbers $z$ with $|z| = z * \bar{z} = 1$.

    Parameters
    ----------
    dim : int
        Dimension of the matrix.
    spec : iterable, optional
        Set of eigenvalues. Must consist of complex roots of unity. Default is None.
    units : iterable, optional
        Complex roots of unity used to build the matrix. Defaults to $\{1, i, -1, -i \}$.
    k : int, optional
        Number of rotations to build the matrix. Defaults to 2 * dim.

    Returns
    -------
    sympy.Matrix
        The generated random unitary matrix.

    See Also
    --------
    random_orthogonal_matrix : Real version of unitary matrices.

    Examples
    --------

    .. ..testsetup::

       >>> from sympy.core.random import rng, seed
       >>> _rng_state = rng.getstate()
       >>> seed(1)

    >>> from sympy import I, pi, exp, expand, simplify, symbols
    >>> from sympy_matrices_random import random_unitary_matrix

    Generate 3×3 unitary matrix (default settings)

    >>> u = random_unitary_matrix(3)
    >>> u
    Matrix([
    [ 0,  0, -I],
    [-1,  0,  0],
    [ 0, -I,  0]])

    >>> expand(u.H * u)
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])

    **Control of Rotations**.
    The **k** argument controls how many elementary complex rotation cells
    are multiplied to form the random matrix. The default is 2*dim.

    >>> u = random_unitary_matrix(5, k=30)  # a denser random unitary matrix

    The **units** argument defines the set of complex numbers
    (roots of unity) from which the elementary rotation cells are drawn.

    >>> roots_of_unity = I, exp(I * pi / 4), -I, exp(-I * pi / 4)

    >>> random_unitary_matrix(3, units=roots_of_unity, k=1)
    Matrix([
    [sqrt(2)*exp(-I*pi/4)/2, 0, -sqrt(2)*exp(-I*pi/4)/2],
    [                     0, 1,                       0],
    [ sqrt(2)*exp(I*pi/4)/2, 0,   sqrt(2)*exp(I*pi/4)/2]])

    Specific eigenvalues may provided as **spec** argument

    >>> x = symbols('x', real=True)
    >>> spec = exp(I * x), exp(I * pi / 4)
    >>> U = random_unitary_matrix(3, spec=spec, units=spec, k=1)

    Without given **spec** the function returns
    a matrix with determinat 1 by default

    >>> random_unitary_matrix(6).det()
    1

    Random determinats can be obtained by providing **spec**

    >>> spec = exp(I * x), exp(I * pi / 4)
    >>> random_unitary_matrix(6, spec=spec).det()
    I*exp(4*I*x)

    .. ..testcleanup::

       >>> assert not rng.getstate() == _rng_state
       >>> rng.setstate(_rng_state)
       >>> assert rng.getstate() == _rng_state

    """
    s = _unitary(dim, units=units, k=k)
    if spec is None:
        return s
    blocks = [_ssample(spec) for _ in range(dim)]  # 1 x 1 complex rotation
    normal_form = Matrix.diag(*blocks)
    return s.H * normal_form * s
