---
name: paper-restructure-vmc-natural
description: "root.tex restructured 2026-07 — attitude VMC added, flat2slope experiments removed, results now rest on natural-terrain experiment; paper is 6 pages"
metadata: 
  node_type: memory
  type: project
  originSessionId: 50f85d68-6558-484d-91a2-3d815091a233
---

On 2026-07-11 root.tex was reworked so the augmented controller (Hopf CPG + attitude virtual model controller) is front-and-center and the flat2slope experiments are gone.

**What changed:**
- Problem Statement: new subsection `sec:vmc` (Attitude Virtual Model Control) modeling the code in `methods/marxefe_optimizer.py` (`JointCPG.step`): kinematic spring-damper on roll (→0) and pitch (→slow EMA baseline), distributed to knees, clipped. Also added the Righetti contact-feedback eq `eq:feedback` and joint map `eq:joint-map` (with VMC term Δ_i).
- Related Work: paragraph on closed-loop CPG / posture feedback / VMC citing [[cpg-no-attitude-feedback]] fix (pratt2001virtual, ajallooeian2013central, PatternGenerators2008).
- Removed flat2slope results entirely (Exp1 single flat→10° transition tab:eventtrigger, stability goal-prior tab:stable, repeated-transitions tab:repeated + fig:eventtrigger/fig:theta/fig:noadapt). Reason: attitude VMC makes 10-15° too easy — see [[attitude-feedback-subsumes-adaptation]] and [[stability-goal-prior-breakthrough]] (that breakthrough was a flat2slope result and is now cut).
- Natural-terrain experiment (`experiment-natural-adapt`, tab:natural) is now **Experiment 1** and the main quantitative result; see [[natural-adapt-dt-findings]]. Exp2 (β-sweep) and Exp3 (target-velocity) remain as placeholders (figures commented out, no framebox).
- Abstract, Discussion, Conclusions rewritten to drop flat/10° numbers.

**Open item:** citation key `eriksson2019turbo` (TuRBO trust-region BO) is still missing from references.bib — pre-existing, user must add via Zotero. `righetti2008pattern` was a wrong key; corrected to `PatternGenerators2008` (the entry that exists).

Compiles at 6 pages (ieeeconf). The old `\addtolength{\textheight}{-12cm}` and empty APPENDIX were removed during compression.
