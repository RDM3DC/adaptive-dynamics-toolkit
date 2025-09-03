"""
Adaptive Dynamics Toolkit (ADT): A unified framework for adaptive computing paradigms.
"""

__version__ = "0.1.0"

from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version

with suppress(PackageNotFoundError):
    __version__ = version("adaptive-dynamics")