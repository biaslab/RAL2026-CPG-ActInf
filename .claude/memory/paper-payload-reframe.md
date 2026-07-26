---
name: paper-payload-reframe
description: "2026-07-25/26 root.tex retitled to \"Event-triggered active inference … sudden payload shifts\", prose compressed throughout, Related Work heading dropped; printing/ holds Bittle CoM-shift hardware"
metadata: 
  node_type: memory
  type: project
  originSessionId: c7e53ce7-85eb-41f9-b57d-5f1001b327a7
  modified: 2026-07-26T10:54:07.439Z
---

**Title.** `root.tex` now reads *"Event-triggered active inference for online gait
adaptation to sudden payload shifts in quadrupedal locomotion"*. Five earlier
candidates are kept commented above it, including the previous title
("Event-triggered learning for adaptive central pattern generation in quadrupedal
robot locomotion"). The framing narrowed from generic operating-condition change to
**payload shift specifically**; the abstract's headline is now that screening
candidates against the fall memory (instead of physically trialing them, as grid
search / BO must) lets the agent learn *within a bout* to stop selecting gaits that
fall.

**Prose pass.** A near-uniform compression of §I–§II — Introduction, CPG description,
VMC section, problem statement — merging sentences and cutting hedges rather than
changing claims. The `\subsection{Related Work}` heading is commented out; that
material now runs inline in the Introduction. The joint-map equation was split into an
aligned two-line form. Funding line now also credits the **NGF AiNed XS Europe grant**
alongside EAISI. Added `zhang2024online` (Zhang, Bellegarda, Shafiee, Ijspeert, *Online
Optimization of Central Pattern Generators for Quadruped Locomotion*, arXiv:2410.16417)
to `references.bib`.

**Hardware.** Untracked `printing/` (2026-07-25) holds the Bittle CoM-shift rig:
`bittle_com_harness.scad/.stl`, `bittle_pinion.scad/.stl`, `bittle_base.stl`,
`bittle_carriage.stl` — a rack-and-pinion carriage to move a payload along the trunk.
This is the start of the hardware validation the simulated review asked for.

Supersedes the structure recorded in [[paper-restructure-vmc-natural]]; code side in
[[esc-baseline-and-readapt]] and [[async-unified-aif-agent]].
