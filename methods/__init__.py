"""CPG optimization methods package.

Portability shims below run on first ``import methods.*`` — i.e. before any
submodule imports torch / pybullet — so they fix two Windows issues without
needing the caller to set environment variables manually:

  * OpenMP duplicate-runtime abort (torch's MKL libiomp vs pybullet).
  * UnicodeEncodeError when printing non-ASCII (e.g. the ✅/° characters in the
    progress logs) on a default cp1252 Windows console.
"""

import os
import sys

# Must be set before torch is imported, otherwise the OpenMP runtimes have
# already been linked and abort with "OMP: Error #15".
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Make stdout/stderr tolerate non-ASCII regardless of the console code page.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

__all__ = ["BOOptimizer"]
