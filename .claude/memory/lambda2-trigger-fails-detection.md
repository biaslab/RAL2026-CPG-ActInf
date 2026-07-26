---
name: lambda2-trigger-fails-detection
description: 100-seed dt run shows the Newton-decrement lambda^2 trigger fails to detect the 10 deg transition; CE ratio is the working trigger
metadata: 
  node_type: memory
  type: project
  originSessionId: 62d0817c-f92c-4295-8ea6-225985c381da
---

The decision-theoretic Newton-decrement trigger (lambda^2 vs control-cost budget tau, `--trigger dt` in experiment-flat2slope-adapt) **fails to detect the 10° transition**. 100-seed run (2026-07-10): median trigger_t = 32.3 s though the robot crosses the slope at ~10 s, fires in only 66% of runs, ~2 s before the fall. All arms collapse to ~49–61% falls (oracle 49, noadapt 60, marxefe 60, grid/bo 61), statistically indistinguishable — the event-triggered story does not reproduce.

Cause (not a trivial bug): lambda^2 = controllable projection of the goal error through the linear MARX control gain B. The lambda^2/tau ratio sits at **~0.1 (10x below threshold) through the whole climb** and only spikes at terminal instability. The linear surrogate reports the flat gait as still first-order optimal on the incline because the fixable error lies in B's null space — exactly the caveat in `notes/prediction-error-triggers.md`, but it bites at 10°, not just >=15°. Consistent with the paper's own Discussion (linear model can't predict which distant params are stable).

By contrast the **CE ratio trigger fires at median 10.7 s (92% of runs)** on the same harness — it keys on the cross-entropy *value*, which does rise on the incline. CE (K=16) is the working detector and the one behind the paper's Table I.

Implication: the lambda^2 harmonization of Methods (`sec:trigger`) + Experiments (`sec:exp-trigger`) is **empirically unsupported**; revert to the CE trigger or present lambda^2 as an analyzed-but-rejected alternative / future work. Do NOT populate the tables with dt numbers.

Note: experiment-flat2slope-adapt is harder than experiment-eventtrigger (oracle falls ~92% here even with CE at 6 seeds), so it is NOT the source of Table I regardless of trigger. Supersedes the harmonization direction in [[eventtrigger-experiment-findings]].
