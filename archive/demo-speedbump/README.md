# Speed-bump adaptation demo

A dashboard-style video (`speedbump_dashboard.mp4`, 1280x720 @ 33 fps) showing
the Laikago walking over a speed bump (10° up-slope, 0.7 m flat top, 10°
down-slope, starting 1.8 m ahead) while a MARX model monitors its own
prediction error and the CPG parameters switch from the flat-terrain optimum
to the sloped-terrain optimum when the bump is detected.

Layout:

- **Left** (full height): zoomed-in tracking render of the robot; amber strips
  on the ground mark the bump's ramp edges.
- **Right top**: running prediction error — the 0.5-s-ahead rollout error of
  the MARX model (thin = raw, bold = moving average). It spikes when the front
  feet hit the ramp; the dashed orange line marks the parameter switch, the
  gray band the time spent on the bump.
- **Right bottom**: 8 sliders showing the live CPG parameter vector
  (positions normalized to the shared bounds in `methods/cpg_bounds.py`);
  thin ticks mark where the sloped-terrain optimum sits.

## Mechanics

- Controller and simulation setup are imported unchanged from
  `methods/marxefe_optimizer.py` (`JointCPG`, `load_environment`, `load_robot`,
  `reset_simulation`); terrain is a `multislope` heightfield from
  `methods/terrain.py`.
- Parameter sets are the seed-0 optima from
  `figures/cpg_optima_by_parameter.csv` (rows `flat,0` and `sloped,0`).
- The MARX agent (`build_marx_agent`, forgetting 0.995) is used purely as a
  predictor — no EFE control. One-step innovations at dt = 10 ms are dominated
  by persistence and barely register the terrain change, so the monitored
  signal is the H = 50-step (0.5 s) rollout error: predictions issued 0.5 s
  earlier (assuming controls stay constant) compared against the observation
  that arrives. Its EMA is tested against a flat-ground baseline
  (mean + 6·std); on the first crossing the parameters ramp to the sloped
  optimum over 0.4 s.

## Run

From the repo root:

```bash
python demo-speedbump/run_demo.py             # simulate (~3 min) + compose video
python demo-speedbump/run_demo.py --sim-only  # simulation only (writes frames/ + demo_data.pkl)
python demo-speedbump/run_demo.py --compose-only   # re-compose video from saved data
python demo-speedbump/run_demo.py --sim-only --no-frames  # quick behavioral check, no rendering
```

Intermediates (`frames/`, `demo_data.pkl`) are git-ignored; the video is the
deliverable.
