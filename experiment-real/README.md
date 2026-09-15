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
shifter_test.py       is the CoM mass shifter plugged in, and on which pin?
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

   **The IMU has two switches and only one is safe to touch.** `gb` toggles the
   firmware's balancing *behaviour* (which has to go, or it fights the CPG for
   the joints) and leaves the IMU sampling; `G` switches the IMU *module* off,
   after which `v` still answers — with the last sample it ever took, frozen,
   until the robot is power-cycled. `PetoiRobot.deacGyro()` sends `G` on
   NyBoard-class firmware, so `BittleLink.connect()` does not call it: it sends
   `gb` itself and then verifies the IMU is actually changing. If a run ever
   aborts with "the IMU is FROZEN", power-cycle the robot — nothing on the
   serial link brings it back.
3. **Harness.** `shifter_test.py --scan` first, if you have just plugged the
   servo in. Two things make "which pin is it on?" harder than it looks, and
   both have already cost a debugging round:
   - The board's numbers are not the joint indices. Per
     [Petoi](https://bittle.petoi.com/4-connect-the-wires/nyboard), "the index
     number of the joint servo has no corresponding relationship with the PWM
     PIN on the main board" — the silkscreen tells you nothing about which
     index to command, so sweep rather than read.
   - Motion at startup is not evidence. Connecting reboots the board and every
     joint snaps to its rest angle before the test runs; only the creep and
     swing phases count, which is why `--scan` sweeps all pins inside one
     connection.

   Nor can the link help: nothing on it distinguishes a connected pin from an
   empty one (the PWM expander has no feedback and the joint readback reports
   what was *commanded*), so the script swings each spare pin a few degrees and
   asks you, while measuring whether the trunk attitude tracks the swing as
   corroboration. It refuses ports 8–15 — those are the gait's own joints, and a
   pin shared between a leg and the harness is double-booked inside the single
   `I` packet each control tick writes (`BittlePhysics` refuses such a
   `--shift-port` outright). If the servo really is in that block, put the robot
   on a stand, find it with `--scan --include-legs` or `--port N --force`, then
   move the lead to a spare pin before any walking run.

   Once the pin is known, `--find-travel` creeps outward one side at a time and
   asks you to call each end stop, then prints the `--shift-centered` /
   `--shift-shifted` pair with a margin — the servo's zero is *not* mid-rack, so
   do this before `--mode shift` cycles the full ramp. No harness fitted?
   `--manual-shift` prompts you to move the payload by hand at each event.
4. **Walk.** The incumbent must actually walk before any arm means anything. If
   the robot walks *backwards*, swap the sign of the hip mapping (this is a known
   failure mode of the simulated optimum — see the `backward-gait-ceiling` note).
   If it barely moves, retune on the robot and save the result as
   `results/incumbent.json` (`{"params": [...8 floats...]}`).

   **Dragging feet / not clearing the ground** is the common one, and the raw
   parameters hide how small the motion is. `JointCPG.step` drives the knee in
   swing only —

   ```python
   knee_angles = self.KNEE_OFFSET - knee_amp * np.maximum(0.0, y_new)
   ```

   — and the limit cycle has radius √U = 1.414, so the simulated incumbent's
   `kneeA` is **8.5° of knee rotation**, a few millimetres of foot lift, before
   any servo droop under the payload. `bi.travel_degrees()` reports it and
   `stand_test.py` prints it every run. Two independent knobs:

   - `--knee-lift DEG` — how far the foot is picked up (the gait vector's
     `kneeA`, in degrees at the joint). This is ground clearance.
   - `--knee-offset DEG` — the stance knee angle, i.e. how extended the legs are
     and how high the trunk rides. This is what a payload squashing the servos
     eats into. The park pose follows it, in both scripts, so the robot does not
     drop to the default crouch whenever it settles or recovers.

   Note that `DKNEE_CLIP` (±4.2°) does **not** scale with the gait, so doubling
   the knee lift halves the attitude correction's authority relative to the
   stride — compensate with `--attitude-gain` if the leveling gets sloppy.

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
* **The control rate is capped by the firmware at ~29-40 Hz, and that is what
  makes the walk look jittery.** Measured on `N_250224` (2026-08-19):

  | path | period | rate |
  |---|---|---|
  | joint write + IMU read | 34.8 ms | 28.8 Hz |
  | joint write alone | 27.7 ms | 36 Hz |
  | `--imu-every 2` (read amortised over two ticks) | 30.9 ms | 32.3 Hz |

  Three things it is *not*, each measured rather than assumed:

  - **Not the packet.** 18 bytes is 1.25 ms at 115200, and `I` is the
    *simultaneous* binary move — ardSerial's skill compiler builds one row for
    `i`/`I`, one row per joint for `m`.
  - **Not the host.** Deleting the vendored `time.sleep(0.01)` in
    `serialWriteByte` and the 1 ms poll in `printSerialMessage` changes the
    period by under a millisecond.
  - **Not the token echo.** Skipping the ack looks like free rate — the board
    swallows joint packets at ~40 Hz when nothing else is asked of it — but a
    `v` sent straight after a binary packet is consumed as payload (11/40 IMU
    reads survived), and the 5 ms settle that fixes it leaves the tick at
    36.1 ms, *slower* than waiting. The IMU read then costs ~31 ms instead of
    ~7 ms: the firmware's per-command work is conserved either way. `--imu-every`,
    which skips whole reads, is the only lever that moves the rate.

  The consequence is mechanical: the servos hold the last commanded angle and
  slew to each new one at ~600°/s, so a tick is a short burst of motion followed
  by a standstill. At the simulated incumbent's 0.366 s stride, 29 Hz gives ~10
  commands per stride and a ~7° step per tick — roughly 12 ms of motion then
  23 ms of stillness. **No filtering removes this**; it is the sample rate
  against the gait frequency. The knobs that help are `--imu-every 2` (rate) and a
  longer stride (`stand_test.py --stride`, which scales `w_swing`/`w_stance`
  together and preserves their ratio). Both scripts warn below
  `TICKS_PER_STRIDE_MIN` = 20 commands per stride.
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
