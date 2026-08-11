# Real-robot experiment — Petoi Bittle

Hardware counterpart of `experiment-simulation/experiment-payload-adapt/`: the
continual **payload-shift adaptation** bout, run on a physical Bittle carrying the
rack-and-pinion CoM harness from `printing/`.

The point of this folder is that the *methods* are not reimplemented here. The
event detector, the responder arms (`noadapt`/`grid`/`bo`/`esc`/`safegp`/`oracle`/
`aif`) and the bout bookkeeping are `methods/continual_driver.py` +
`methods/event_responders.py` — the same code the simulation runs. Only the
physics is swapped:

| | simulation | hardware |
|---|---|---|
| controller | `methods.cpg_controller.JointCPG` | `BittleCPG` (same class, Bittle joint mapping) |
| step | `p.stepSimulation()` | serial write to the servos |
| payload shift | `p.changeConstraint(...)` | harness servo, `frac` → servo angle |
| attitude | `getEulerFromQuaternion` | onboard IMU (`v` command) |
| fall reset | `resetBasePositionAndOrientation` | the operator |

## Files

```
run_experiment.py     the experiment + the bring-up modes (--mode)
stand_test.py         CPG + attitude loop on a STAND: no event, no fall logic
bittle_interface.py   BittleCPG, BittleLink (serial), BittlePhysics (driver contract)
petoi_Hopf.py         the original hand-tuned open-loop CPG demo (the calibration anchor)
*Example.py           vendored Petoi examples
PetoiRobot/           vendored Petoi Python API — do not modify
results/              continual_events.csv, continual_summary.csv, logs/*.npz
```

Scripts add their own directory to `sys.path`, so they can be launched from
anywhere; the vendored `*Example.py` files still need `cd experiment-real`.

## Bring-up (in order, on a charged battery)

Every step below has a `--mode`. Do not skip them: three constants in
`bittle_interface.py` (IMU signs, harness end stops, control rate) are *guesses*
until measured on your robot.

```bash
python run_experiment.py --mode rate     # 1. achievable control rate  -> pick --dt
python run_experiment.py --mode imu      # 2. IMU units and roll/pitch SIGNS
python run_experiment.py --mode shift    # 3. harness servo end stops
python run_experiment.py --mode walk --duration 20   # 4. does the incumbent walk?
```

1. **Rate.** The serial round trip, not the CPG, sets the control rate (~40–70 Hz
   in practice). The CPG integrates at `--dt`, so if `--dt` is faster than the
   link can sustain, the gait runs *slower* than commanded. Round the reported
   median up and pass it as `--dt`; `--imu-every 2` buys rate at the cost of a
   staler attitude reading.
2. **IMU.** Tilt the robot and check the printed signs: `roll > 0` right side
   down (banking right), `pitch > 0` **nose up** — that is the convention the
   attitude gains were validated in (`get_observation` unpacks the simulated
   euler angles as `pitch, roll, yaw` with the robot walking in +Y). Prefer
   `stand_test.py --mode sign`, which does this against a level reference and
   prints a verdict. If either is inverted, run everything afterwards with
   `--roll-sign -1` / `--pitch-sign -1` — the VMC attitude feedback pushes the
   robot *over* with the wrong sign. If no values appear at all, the firmware may
   silence the IMU when gyro balancing is off: retry with `--keep-gyro`.
3. **Harness.** Verify the slug reaches both ends without the servo stalling, then
   set `--shift-centered` / `--shift-shifted`. No harness fitted? `--manual-shift`
   prompts you to move the payload by hand at each event.
4. **Walk.** The incumbent must actually walk before any arm means anything. If
   the robot walks *backwards*, swap the sign of the hip mapping (this is a known
   failure mode of the simulated optimum — see the `backward-gait-ceiling` note).
   If it barely moves, retune on the robot and save the result as
   `results/incumbent.json` (`{"params": [...8 floats...]}`).

### 4b. On the stand, before the floor

`stand_test.py` runs the same control loop as a bout — `BittleCPG.control_tick`
sub-stepped at `--cpg-dt`, plus the VMC attitude correction — with the payload
harness, the detector, the responders and the fall logic all removed, so the only
things under test are the oscillators, the joint mapping and the attitude
feedback. Feet off the ground, so a wrong sign or a hot gain costs nothing:

```bash
python stand_test.py --mode sign                     # IMU direction, guided (do this first)
python stand_test.py --mode still --duration 30      # only the posture correction moves the knees
python stand_test.py --mode walk  --duration 30      # gait + correction together
python stand_test.py --mode still --inject roll      # synthetic attitude: no hands needed
python stand_test.py --mode walk --no-attitude       # open-loop reference
```

`--mode sign` is the one step nothing else can replace: every other check
regresses the knee correction against the attitude the controller was *handed*,
so an inverted IMU passes them all and only shows up when the robot goes over on
the floor. It holds the robot in two known attitudes against a level reference,
reports the response and the cross-axis leak (which catches a firmware that does
not order the `v` token as yaw/pitch/roll), and names the flag to fix.

Each run writes `results/standtest_<mode>_<stamp>.npz` with the per-tick trace
(attitude in and measured, per-leg correction, commanded angles, CPG state) and
prints the achieved control rate, joint travel against the safety limits, how
often the correction saturated its ±4.2° clip, and how often it fell below the
servos' 1° quantum.

Then a first supervised bout, and a session:

```bash
python run_experiment.py --arms noadapt --seeds 1 --duration 90
python run_experiment.py --arms noadapt aif safegp bo --seeds 3 --duration 120
```

Bouts run one at a time and wait for you between them. A fall pauses the run until
you stand the robot up and press ENTER (`--recover auto` uses the firmware's
self-right skill instead). `results/continual_events.csv` is **appended** across
sessions, so an interrupted session loses only the bout in progress.

## Rehearsing without a robot

```bash
python run_experiment.py --dry-run --dry-speed 4 --no-prompt \
    --arms noadapt safegp --seeds 1 --duration 120
```

`--dry-run` swaps in a null serial transport plus a synthetic body attitude, which
exercises the whole chain — detect → request → propose → apply → fall → reset →
re-arm — including the CSV/npz output. **Its numbers are meaningless**: the
synthetic robot is a caricature (`SyntheticRobot`), not a simulator. Use it to
check plumbing and CLI wiring, never to produce a result.

## Hardware limitations, stated plainly

These are properties of the robot, not of the method, and they are the things a
reviewer will ask about:

* **No foot-contact sensors.** The Righetti STOP/FAST feedback needs a per-leg
  contact bit. By default (`--contacts phase`) the CPG is fed the contact pattern
  its own oscillator phase expects, which reproduces the nominal simulated
  behaviour but removes the disturbance-feedback path that exists in simulation.
* **No odometry.** Nothing measures forward speed, so the detector's
  speed-deficit term has nothing to feed on. With `--vx-source none` (default) a
  constant nominal speed is reported, which zeroes that term and leaves a
  **tilt-only** CUSUM. Consequently the `dist` / `trial_dist` columns are
  dead-reckoned from that constant and are **not measurements** — report falls,
  not distance, unless you add external tracking.
* **Attitude-feedback gains are transferred, not tuned.** They are the simulated
  gains rescaled by the knee mapping (`--attitude-gain` scales them, `--no-attitude`
  disables the loop). Expect to tune them on the bench.
* **The oracle arm needs a hardware fit.** The simulated optimum is not an oracle
  for this robot; the arm refuses to run until `results/payload_optima.json`
  exists.
* **Bout length is limited by the arena**, not by the script: the robot walks away
  from its start position and nothing recentres it.

## Joint mapping

The 8-D gait vector is the simulator's (`methods/cpg_bounds.py`), in Laikago
radians. `BittleCPG` rescales it to Bittle degrees with factors anchored so the
simulated flat-optimal incumbent reproduces the hand-tuned gait in `petoi_Hopf.py`:

| | incumbent (sim) | × scale | Bittle | `petoi_Hopf.py` |
|---|---|---|---|---|
| hip amplitude | 0.10 | 120 deg/unit | 12 deg | 12 deg |
| knee amplitude | 0.50 | 12 deg/unit | 6 deg | 6 deg |
| hip offset | — | — | 40 deg | 40 deg |
| knee offset | — | — | 30 deg | 30 deg |
| `w_swing` / `w_stance` | 13.0 / 25.0 | — | — | 12 / 24 |

Leg order differs between the two: the simulator uses `[FL, FR, RL, RR]` while
Petoi indexes clockwise from the front-left (`8` FL, `9` FR, `10` **RR**, `11`
**RL**), so `HIP_PORTS = [8, 9, 11, 10]`. Getting this wrong silently breaks the
trot's diagonal pairing.
