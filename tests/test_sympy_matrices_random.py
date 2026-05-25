import random as system_random

from sympy import (eye, cartes, I, conjugate, cos, sin, symbols, expand,
                   simplify, primefactors)
from sympy.core.random import seed
from sympy.core.numbers import Number
from sympy_matrices_random import (_ssample, _jspec, _ELEMENTARY_SCALARS,
                                   _ELEMENTARY_UNITS, random_matrix,
                                   random_orthogonal_matrix,
                                   random_unitary_matrix)
from sympy.testing.pytest import raises


TEST_DIMS = dict((d, tuple(cartes(range(d), range(d)))) for d in range(2, 6))
TEST_PRECISION = 7
TEST_EPSILON = 1e-7

phi, psi, z = symbols('phi psi z')

# set fixed sympy random seed for testing purposes
sympy_seed = 12
seed(sympy_seed)

# set system random number generator with fixed seed for testing purposes
system_seed = 111
RANDOM = system_random.Random(system_seed)


# === validation functions ===


def _is_zeros(m, precision=None):
    if precision is None:
        return all(x == 0 for x in simplify(m))
    else:
        return all(abs(x) < precision for x in m.evalf())


def _is_eye(m, precision=None):
    return _is_zeros(m - eye(*m.shape), precision)


def _is_isometry(m, precision=None):
    return _is_eye(m.H * m, precision)


def _is_triangular(m, k=0, precision=None):
    s, t = m.shape
    if precision is None:
        return all(m[i, j] == 0 for i in range(s) for j in range(i + k))
    else:
        n = m.evalf()
        return all(
            abs(n[i, j]) < precision for i in range(s) for j in range(i + k))


def _is_diagonal(m, k=0, precision=None):
    return (_is_triangular(m, k=k, precision=precision) and
            _is_triangular(m.T, k=k, precision=precision))


def _is_jordan(m, precision=None):
    jd = set(m.diagonal(1)) in ({0}, {0, 1}, {1}) if m.shape[0] > 1 else True
    return (_is_triangular(m, precision=precision) and
            _is_triangular(m.T, k=-1, precision=precision) and jd)


# === matrix helper functions ===


def _random_matrix(dim, spec=None, scalars=None, units=None, rank=None,
            triangular=False, k=None):
    """generic matrix test function"""
    m = random_matrix(dim, spec=spec, scalars=scalars, units=units, rank=rank,
               triangular=triangular, k=k)
    m = expand(m)

    # test shape
    assert len(m.shape) == 2
    assert m.shape[0] == m.shape[-1]

    # test dim
    assert dim == m.shape[0]

    # test triangular
    if triangular:
        assert _is_triangular(m)

    # test rank
    if rank is None or rank == dim:
        assert dim == m.rank()
        assert _is_eye(m.inv() * m)
    else:
        assert rank == m.rank()

    # test values
    if (all(isinstance(v, (int, Number))
            for v in scalars or _ELEMENTARY_SCALARS) and
        all(isinstance(v, (int, Number)) for v in units or _ELEMENTARY_UNITS)):
        for v in m:
            assert isinstance(v, (int, Number))

    # test eigenvalues
    for ev in m.eigenvals(multiple=True):
        if spec is None:
            if isinstance(ev, int):
                for p in primefactors(ev):
                    assert p in (units or _ELEMENTARY_UNITS)
        else:
            if rank is None or rank == dim:
                assert ev in spec
            else:
                assert ev in spec or ev == 0

    # test the jordan form
    if spec and k == 0:
        assert _is_jordan(m), m
        jspec = _jspec(spec)
        if max(j[0] for j in jspec) == 1:
            assert _is_diagonal(m)
            if len(jspec) == 1:
                v = jspec[0][-1]
                if rank is None or rank == dim:
                    assert list(m.diagonal()) == [v] * dim
                else:
                    assert sum(m.diagonal()) == v * rank
    return m


def _orthogonal(dim, spec=None, *, angles=None, k=None):

    m = random_orthogonal_matrix(dim, angles=angles, k=k)
    m = expand(m)

    assert expand(m.T * m) == eye(dim)
    assert m.det() == 1

    assert len(m.shape) == 2
    assert m.shape[0] == m.shape[-1]

    if dim:
        assert dim == m.shape[0]

    if spec is not None:
        _spec = [cos(s) + I * sin(s) for s in spec]
        _spec += [conjugate(s) for s in _spec]
        for ev in m.eigenvals(multiple=True):
            assert expand(ev) in _spec

    if k == 0:
        # isometry normal form
        assert _is_diagonal(m, 2)

    return m


def _unitary(dim, spec=None, *, units=None, k=None):

    m = random_unitary_matrix(dim, units=units, k=k)
    m = expand(m)

    assert expand(m.H * m) == eye(dim)
    assert abs(m.det()) == 1

    assert len(m.shape) == 2
    assert m.shape[0] == m.shape[-1]

    if dim:
        assert dim == m.shape[0]

    if spec is not None:
        for ev in m.eigenvals(multiple=True):
            assert expand(ev) in spec

    if k == 0:
        # isometry normal form
        assert _is_diagonal(m, 1)

    return m


# === tests ===


def test_elementary():
    for d in TEST_DIMS:
        m = _random_matrix(d, scalars=[2, 3, 5], units=[1], k=1) - eye(d)
        assert sum(m) in (0, 2, 3, 5)


def test_diagonal():
    for d in TEST_DIMS:
        m = _random_matrix(d, spec=[1], k=0)
        assert _is_eye(m)

        m = _random_matrix(d, scalars=[0], units=[1])
        assert _is_eye(m)

        m = _random_matrix(d, spec=[0], rank=0, k=0)
        assert _is_zeros(m)

        m = _random_matrix(d, spec=[1], rank=0, k=0)
        assert _is_zeros(m)

        m = _random_matrix(d, rank=0)
        assert _is_zeros(m)

        m = _random_matrix(d, spec=[1, 2, 3], k=0)
        assert _is_diagonal(m)


def test_jspec():
    def _jspec_args(spec):
        r = []
        for i, v in spec:
            # build random argument for _jspec
            if _ssample([0, 0, 1, 1, 1]):
                if len(r) and r[-1] == v:
                    r.append(None)
                r.extend([v] * i)  # e.g. v,v,v,v,v, None
            else:
                r.append((i, v))  # e.g. (5, v)
        return r

    a, b = symbols('a b')
    spec = 1, 2, 2, a, b, I, I, I
    num = 1, 2, 3, 4, 5
    for _ in range(1_000):
        args = [(_ssample(num), _ssample(spec)) for _ in range(10)]
        args_in = _jspec_args(args)
        args_out = _jspec(args_in)
        assert args == args_out


def test_jordan():
    for d in TEST_DIMS:
        m = _random_matrix(d, spec=[2, 2, 3], k=0)
        assert _is_jordan(m)
        m = _random_matrix(d, spec=[2, 2, None, 2, 2, 2, None, 2], k=0)
        assert _is_jordan(m)


def test_rank():
    for d in TEST_DIMS:
        for r in range(1, d):
            m = _random_matrix(d, rank=r)
            assert r == m.rank()
            m = _random_matrix(d, spec=[2, 2, None, 2, 2, 2, None, 2], rank=r)
            assert r == m.rank()


def test_idempotent():
    for d in TEST_DIMS:
        for r in range(1, d):
            m = _random_matrix(d, spec=[1], scalars=[1], units=[1], rank=r)
            assert m * m == m


def test_nilpotent():
    for d in TEST_DIMS:
        for r in range(1, d):
            m = _random_matrix(d, spec=[0, 1], scalars=[1], units=[1], rank=r)
            assert m.rank() == r


def test_raises():
    with raises(RuntimeError):
        # fails to find 2 x 2 Jordan blocks with rank 1
        random_matrix(dim=2, spec=[0], rank=1, k=0)


def _test_orthogonal():
    for d in TEST_DIMS:
        i = _orthogonal(d, spec=[0], k=0)
        assert _is_eye(i)

        i = _orthogonal(d, angles=[0])
        assert _is_eye(i)

        m = _orthogonal(d)
        assert _is_isometry(m)
        assert m.det() == 1

        m = _orthogonal(d, spec=[phi])
        assert _is_isometry(m)

        m = _orthogonal(d, angles=[phi])
        assert _is_isometry(m)


def _test_unitary():
    for d in TEST_DIMS:
        i = _unitary(d, spec=[1], k=0)
        assert _is_eye(i)

        i = _unitary(d, units=[1])
        assert _is_eye(i)

        m = _unitary(d)
        assert _is_isometry(m)
        assert m.det() == 1

        m = _unitary(d, spec=[z])
        assert _is_isometry(m)

        m = _unitary(d, units=[z])
        assert _is_isometry(m)
