---
name: prefers-notebooks-for-analysis
description: "User wants results-analysis code delivered as Jupyter notebooks (.ipynb), not .py scripts"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 442f0b08-3945-4276-8849-c8856d320d5e
---

The user prefers results analysis / figure generation to live in **Jupyter
notebooks (.ipynb)**, not standalone `.py` scripts.

**Why:** notebooks let them re-run cells, tweak plots interactively, and view
figures inline — the normal workflow for iterating on paper figures.

**How to apply:** when building analysis/plotting deliverables, ship an `analyze.ipynb`
(or similar) rather than `analyze.py`. Keep reusable heavy logic (data loading,
palette, figure functions) in a shared `methods/*.py` module and have the notebook
import and call it, so the notebook stays thin and the logic is testable. Experiment
RUN scripts (`run_experiment.py`) stay as `.py`; this is about the analysis side.
See [[consolidated-continual-experiments]].
