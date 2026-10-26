"""Non-destructive replication pipeline helpers.

This package deliberately contains no acquisition, encoder extraction, adaptation,
or batch-launch code.  Those actions remain user-owned.
"""

from .pipeline import plan_cell, run_cell

__all__ = ("plan_cell", "run_cell")
