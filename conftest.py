"""Ensures the repo root is on ``sys.path`` so ``tests/`` can import ``core``.

An (otherwise empty) ``conftest.py`` at the repository root makes pytest add
this directory to ``sys.path``, regardless of the current working directory
pytest is invoked from. Without it, test modules under ``tests/`` (which
intentionally has no ``__init__.py``, so it isn't itself a package) would not
reliably be able to ``import core`` or ``import views``.
"""
