---
name: payload-shift-experiment
description: experiment-simulation/experiment-payload-adapt — 8 kg payload shifting 0.20 m lat+back at T/2; screened regime map; persistent-mismatch challenge replacing terrain transitions
metadata: 
  node_type: memory
  type: project
  originSessionId: aece00c0-2916-4a97-8f01-c7a9ab6d2bbd
---

Created 2026-07-14: `experiment-simulation/experiment-payload-adapt/` — flat ground, 8 kg box
constraint-fixed 0.15 m above the trunk, shifts 0.20 m lateral + 0.20 m rearward
at T/2 (ramped 1 s). Rationale: terrain transitions failed because falls were
transition shocks ([[oracle-cannot-beat-noadapt]]); a CoM offset is a
*persistent* mismatch hitting the attitude channel
([[stability-goal-prior-breakthrough]]).

Screened regime map (incumbent = flat-optimal, feedback ON):
- 4 kg/0.15 & 6 kg/0.20: absorbed by VMC feedback, no trigger (ratio ~1.1).
- **8 kg/0.20 (default)**: no-adapt survives but limps (vx ~0.27 vs 0.74
  achievable by another gait under same load) → J/velocity/power gap, CE
  trigger fires 1–4 s post-shift and re-fires.
- 8 kg/0.225: falls 1/3 seeds. 8 kg/0.25: incumbent falls 3/3; only
  backward-walking gaits survive → likely unrecoverable, avoid.
- 10 kg/0.30: falls during ramp (pure shock).

Pitfalls found: (1) never attach the payload pre-shifted at rest — crouched
settle under 0.25 m offset tips statically; fit oracles by shifting at t=1 s
while walking (fit_payload_oracles.py does this). (2) changeConstraint can only
move the CHILD pivot; create the box at the shifted spot for shifted-start.

Protocol: fit_payload_oracles.py (BO per condition + prints cross-penalty
go/no-go gap) → run_experiment.py (arms noadapt/grid/bo/marxefe/oracle,
per-phase falls + vx + tip + power/CoT in manifest) → analyze.py.
Machinery imported from experiment-flat2slope-adapt (single source of truth).

2026-07-14 cross-penalty screen: **GO, gap = +1.38** (first challenge to pass).
V matrix (6 seeds): incumbent +0.946/−0.206 (centered/shifted); centered_opt
+0.979/**−1.417** (phase-1 optimum is catastrophic in phase 2 — the missing
precondition finally holds); shifted_opt +0.886/−0.037. payload_optima.json
saved.

2026-07-14 first run (10 seeds, cusum, 30 s, noadapt/marxefe/oracle): **oracle
beats no-adapt on EVERY seed** (vx₂ 0.53 vs 0.37, dist 15.8 vs 13.8 m, tip 2.85°
vs 3.35°; same CoT ~262 J/m so oracle strictly better in J); zero falls at 0.20
offset (limp regime as screened). CUSUM latency 1.4–3.3 s, re-fires. **MARX-EFE
= no-adapt exactly**: adapts 100% of post-trigger windows but proposals move
only 0.4% of param range (max 3.2%) while shifted_opt is 54–78% away on
coupling/w_swing/F_FAST/STOP_GAIN → incumbent-pinning
([[track1-marxefe-no-adaptation]]), bottleneck now the METHOD not the
environment.

2026-07-14 loosening experiment: neither prior unpins MARX-EFE. New knobs
`--marx-prior-scale` (MARX_CONTROL_PRIOR_SCALE, f2s module attr) and
`--marx-vel-std` (MARX_GOAL_VEL_STD, selection-only, monitors keep MON_VEL_STD).
Probes: scale 1.0/3.0 alone AND +vel-std 0.2 all give ~2% range movement,
vx₂≈0.36 — pinning survives → root cause is IDENTIFIABILITY: MARX posterior
only has data at the incumbent, input-coefficient block ~0, EFE flat in u,
epistemic term too weak to break the loop. 10-seed run (scale 3.0, vstd 0.2):
marxefe 4/10 falls (was 0/10 pinned), vx₂ 0.36 unchanged — loosening strictly
hurts. BO: **9/10 falls** (p<0.001), moves 17%/93% mean/max of range, P₂ 152 W —
destructive exploration, safeguard can't prevent within-window falls. Nobody
captures the oracle gap (0.53) online: model-based can't extrapolate,
model-free falls while exploring. Pinned marxefe archived in
runs/archive-marxefe-scale0.15/; manifest now merges on re-run.

2026-07-14 sysid arm (`marxefe_sysid`, user idea "train on first load"): OU
dither (std 5% range, tau 0.5 s, from t=4 s until trigger; decays smoothly
after — all changes ramped, never instantaneous) + forgetting 0.999 (10 s
memory) identifies the u→y map in phase 1. RESULT LADDER: identification
unpins the EFE (proposals 2%→40-50% range) BUT scale 3.0 slams to box bounds
and falls 4/4 (linear model extrapolates); **scale 0.5 = trust region: 1/10
falls, 9 incremental proposals to ~58% range**. Remaining failure: moves are
SIDEWAYS — dist-to-shifted-opt 0.304(inc)→0.282(first)→0.280(last); late-ph2
vx 0.18 < noadapt 0.24 (oracle 0.32). Phase-1 (centered) model doesn't
transfer: the u→y map itself changes at the shift. Next rung: keep gentle
excitation ON post-shift so the model re-identifies under the new dynamics,
or repertoire/library switching (oracle IS a library switch; Cully 2015).
Flags: --marx-forgetting, --train-{t0,std,tau} (renamed from sysid 2026-07-15:
training is part of the `marxefe` arm now, --no-train ablates; payload-exp
defaults = scale 0.5 / vstd 0.2 / forget 0.999).

2026-07-15 FULL 60 s run @ 8 kg, lat 0.225 / back 0.20 (10 seeds, cusum, refit
optima; old results in archive-lat0.20-30s/): cross-screen gap +0.86 GO, but
the refit shifted optimum is a near-stationary BRACE (V=-0.07, vx~-0.1, 182 W).
Result: noadapt 10/10 falls, survives 13.5±6.3 s post-shift; grid/bo/marxefe
10/10 falls at 8.1-8.5 s — **every online adapter falls FASTER than no-adapt**
(trying to keep/raise speed at the cliff is fatal); oracle 0/10 falls by
bracing (vx -0.15). Detection is solved (latency 1.5-1.7 s all arms). ROOT
CAUSE: the response objective rewards forward velocity (windowed J's r_v, EFE
vel goal vstd 0.2) exactly when survival requires sacrificing it. At the cliff
the correct policy is a brace reflex: tight ATTITUDE goal + LOOSE velocity goal
(inverts the limp-regime tuning; cf [[stability-goal-prior-breakthrough]]).
Next: regime-dependent goal prior (loosen vel / tighten attitude on trigger),
or accept 0.20 limp regime as the marxefe showcase and 0.225 as the oracle-gap
demonstration.

2026-07-15 offset boundary (in-place sweep, 10 seeds each; NOTE this clobbered
the manifest — grid/bo stayed at 0.225 while noadapt/marxefe/oracle went to
0.21 then 0.215; 0.21 data lost. NPZ filenames don't encode offset. Fix: encode
offset in filename before future sweeps): 0.21 noadapt 2/10 & marxefe 2/10 (both
survivable limp); 0.215 noadapt 6/10 marxefe 9/10 (adaptation turns fatal);
0.225 all 10/10. Boundary where velocity-chasing flips lethal ~0.215.

2026-07-15 PRIOR SWEEP stage 1 (marxefe @0.215, 6 seeds, 40 s, scratchpad CSV
sweep_priors_0215.csv, NOT manifest): control-prior SCALE is the dominant safety
lever — scale 1.0 catastrophic (6/6 falls, ~3 s, tip 22-27°: overshoots to bad
params, maxdev 0.78); default 0.5 too loose; 0.3 = safe trust region (maxdev
0.32). Velocity-goal looseness is the secondary lever and only helps inside a
tight trust region: best cell scale 0.3 + vel_std 3.0 = 2/6 (beats noadapt 4/6),
but agent still walks (vx +0.28) not braces (oracle -0.1), win marginal. tip2
tracks falls tightly → attitude is the untested axis. Added MARX_ROLL_DEG/
MARX_PITCH_DEG constants (isolated from monitors, like MARX_GOAL_VEL_STD). Stage
2 running: scale{0.2,0.3} x vel_std{3,10} x roll_deg{6,3,1.5}, 10 seeds.

Stage 2 DONE (10 seeds, sweep_priors2_0215.csv; added MARX_ROLL_DEG/
MARX_PITCH_DEG constants to f2s, isolated from monitors): **NEGATIVE RESULT —
prior tuning does NOT fix MARX-EFE at the 0.215 cliff.** noadapt 7/10, oracle
0/10 (p=0.003, and at 0.215 the 0.225-fit optima WALK forward vx+0.22, not
brace — a good reachable gait exists). Best marxefe cell s0.3/v10/r6 = 3/10 but
p=0.18 (not significant); all 12 cells scatter 3-8/10, no monotone trend in
vel_std OR roll_deg; agent keeps walking (vx~+0.25, tip~5) never braces.
CRUCIAL: stage-1 "winner" s0.3/v3 (2/6) → 8/10 at n=10 = it was small-n NOISE
(I over-read it). Only robust large effect = scale 1.0 overshoot catastrophe.
DIAGNOSIS: bottleneck is the MODEL not the prior — agent trained only on
centered-load dynamics, u→y map changes at shift, EFE gradient points wrong;
oracle proves the safe gait exists+reachable but MARX can't find it from
pre-shift training. NEXT (recommended, not priors): keep small excitation ON
post-trigger so model RE-IDENTIFIES under shifted load (~5-line change: drop the
`trigger_step is None` guard in the train-dither block), then EFE acts on the
corrected model. Fallback framing: 0.20=limp regime marxefe works; 0.215+=cliff
where pre-shift model-based adaptation provably can't reach the safe gait →
motivates online re-ID / gait library.
