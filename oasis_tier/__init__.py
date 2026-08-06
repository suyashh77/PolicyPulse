"""Tier 2 — LLM-driven simulation on OASIS (CAMEL-AI).

Tier 1 (`core/`) is a fast numeric engine: 0.55s per run, free, bit-reproducible.
It is the workhorse for parameter sweeps, calibration, and CI.

Tier 2 is this package: real LLM agents that read the announcement *text*,
write posts, comment, like, repost, and follow each other on a simulated Reddit.
It is slow and costs money, so it is the microscope — used to check that Tier 1's
abstraction holds, and to answer questions Tier 1 structurally cannot:

  - Does the *wording* of the announcement change the reaction?
  - What happens if the brand posts a clarification on day 9?
  - What do people actually say?

Both tiers emit the same report shape, so the reporting layer and UI work
against either.

OASIS requires Python >=3.10,<3.12 and is not installed by default. Imports are
guarded so this package stays importable on the project's 3.12 interpreter;
calling into it without OASIS raises with instructions.

    conda create -n policypulse-oasis python=3.11 -y
    conda activate policypulse-oasis
    pip install -r requirements-oasis.txt
"""
from __future__ import annotations

OASIS_AVAILABLE = False
OASIS_IMPORT_ERROR: str | None = None

try:  # pragma: no cover - depends on optional install
    import oasis  # noqa: F401

    OASIS_AVAILABLE = True
except Exception as exc:  # ImportError, or a transitive dependency conflict
    OASIS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def require_oasis() -> None:
    """Raise a useful error if the Tier-2 dependencies are missing."""
    if not OASIS_AVAILABLE:
        raise ImportError(
            "OASIS is not available in this interpreter.\n"
            f"  underlying error: {OASIS_IMPORT_ERROR}\n"
            "  OASIS needs Python >=3.10,<3.12. Set it up with:\n"
            "    conda create -n policypulse-oasis python=3.11 -y\n"
            "    conda activate policypulse-oasis\n"
            "    pip install -r requirements-oasis.txt"
        )
