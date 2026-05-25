# SymPy Matrices Random

A Python package for working with random matrices using SymPy.

## Installation

```bash
uv add sympy-matrices-random
```

## Development Installation

```bash
git clone https://github.com/yourusername/sympy-matrices-random.git
cd sympy-matrices-random
uv sync
```

## Building Documentation

```bash
cd docs
make html
```

The documentation will be generated in `docs/_build/html/`.

## Running Tests

```bash
uv run pytest
```

## API Usage

```python
from sympy_matrices_random import random_matrix

# Create a random 3x3 matrix
matrix = random_matrix(3)
```

## License

MIT License