"""Deprecated compatibility package for legacy script imports and commands.

New code must import from :mod:`glucose_forecasting`. The ``scripts`` package
remains available during the migration and is scheduled for removal in the next
major release; see ``docs/LEGACY_API.md``.
"""

__deprecated__ = (
    "Use the glucose_forecasting package and glucose command for new code. "
    "The scripts package is scheduled for removal in the next major release."
)
