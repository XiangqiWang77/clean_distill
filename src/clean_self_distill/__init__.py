"""Clean Self-Distillation (CSD).

The package deliberately keeps the target answer outside the specialization
API.  A temporary teacher is built only from a target-derived, sanitized skill
card and independently proposed/solved candidates.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
