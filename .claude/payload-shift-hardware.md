# Payload shift on hardware: what the rod does after a fall

Why the CoM harness rod misbehaves on the Bittle, and which parts are fixed.
Sources: `methods/continual_driver.py`, `methods/continual_driver_aif.py`,
`experiment-real/bittle_interface.py`, `experiment-real/run_experiment.py`,
`experiment-simulation/experiment-payload-adapt/run_experiment.py`.

## What drives the rod

The driver keeps one scalar, `shift_frac` in $[0,1]$
(`continual_driver.py:151`), ramped toward 1 while the event is engaged and
toward 0 while healthy, at `DT / event_ramp_t` per tick (`:197-200`). Hardware
maps it linearly onto a servo angle,

$$\text{deg}(f) = \text{centered} + f\,(\text{shifted} - \text{centered})$$

in `BittlePhysics._shift_deg`, re-commanded **every control tick** inside the
same `I` packet as the eight leg angles (`bittle_interface.py:843-845`). The
simulation does the same thing with `p.changeConstraint` every step
(`experiment-payload-adapt/run_experiment.py:147-149`).

So anything that writes that servo port besides the control loop shows up as a
fight at tick rate.

## Hammering: check the calibration pair FIRST

Observed on ARGENTUM 2026-09-15: the rod hammers against its mechanical guard,
starting not at the recovery but 4-10 s later when the event re-engages, and
travelling **clockwise when it should go counter-clockwise**.

That combination is a calibration fault, not a software fight. The command
stream for the harness port is monotone by construction (`_shift_deg` is linear
in `frac`, `frac` is clipped to $[0,1]$, and `CommandQuantizer` only ever holds
or advances), so the loop cannot oscillate on its own. A rod driven past its end
stop stalls there and buzzes, and nothing on the serial link reports a stall.

The trap was the defaults: `--shift-centered` was `0.0` and `--shift-shifted`
`60.0`, which are placeholders rather than a measurement. A bout launched
without both flags drove the rack `0 -> +60`, the opposite direction from a
robot calibrated `30 -> -60`, and past the end of the rack. **The startup banner
says which pair was used** -- `shift: servo 0: 0 -> 60 deg, ramp 1s, ...` -- so
read that line before theorising.

Fixed 2026-09-15 in `run_experiment.py`:

- `--shift-centered` / `--shift-shifted` no longer default. `require_shift_calibration()`
  runs immediately after `parse_args()`, before anything opens the serial port,
  and refuses `--mode run` or `--mode shift` on the robot without both.
  `--dry-run` and `--manual-shift` still fall back to the old placeholders.
- Each bout now reports the travel actually commanded, to be compared against
  what `--find-travel` printed:
  `harness commanded +0 .. +60 deg over 3000 ticks (configured +0 -> +60)`,
  plus a warning when the harness never moved at all (wrong `--shift-port`).

If the banner shows the right pair and it still hammers, the horn seating is
inverted for this robot: swap the pair, or re-seat the horn so the servo zero
sits at one end of the rack. Only then is the two-writer theory below worth
revisiting.

## Fixed 2026-09-15: the recovery re-enabled the firmware's balancing

**Symptom this was chasing.** Rod misbehaving after a recovery. NOTE: this was
not the cause of the observed hammering (see the section above); it is a real
latent bug that was found while looking for it, and fixing it did not stop the
hammering on ARGENTUM.

**Cause.** `BittlePhysics.reset()` fires firmware skills on every recovery:
`link.posture(recover_skill)` under `--recover auto`, and `link.posture("balance")`
unconditionally. Those switch the firmware's IMU balancing back ON, which is
exactly what `BittleLink.connect()` turns off with `gb` because, in its own
words, "the firmware's own balancing fights the CPG for the joints". Nothing
re-asserted it, so one fall poisoned the rest of the bout. The harness shows it
worst because `SHIFT_PORT = 0` is a head/spare index the firmware also drives;
the guard in `BittlePhysics.__init__` only refuses the leg ports 8-15.

**Fix.** `reset()` now calls `link.disable_gyro_balance()` after the posture
calls, guarded by `keep_gyro`. `gb` is a toggle, but `disable_gyro_balance()`
reads the ack (`g` = off, `G` = on) and re-sends, so repeated calls are safe. A
`dry_run` early-return keeps the rehearsal quiet.

**Diagnostic for next time.** Send `gb` after a recovery and read the ack: `g`
means balancing had been on and you just switched it off; `G` means it was
already off and the fight is coming from somewhere else. Any new
`link.posture(...)` call needs the same re-assert after it.

## Fixed 2026-09-15: `neutral_pose` parked the wrong servo

`BittleLink.neutral_pose()` is a `@staticmethod` that hard-coded the module
constant `SHIFT_PORT`, while `actuate()` used `self.shift_port`. So `_park()`,
the "recentre the harness" step of both `reset()` and `disconnect()`, always
wrote port 0 regardless of `--shift-port`: on any other port the harness was
never actually parked, and port 0 was driven to the harness angle instead.
`neutral_pose` now takes a `shift_port` argument; `_park()` and the `--mode shift`
sweep in `run_experiment.py` pass the real one. Callers that omit it (the
`stand_test.py` homing, `measure_period`) keep the module default.

## Still open: the event recurs after every fall, by design

There is no single-event mode. The event engages once and persists, but the fall
branch reschedules it: `next_event_t = t + rng.uniform(gap_min, gap_max)`
(`continual_driver.py:298`), and that is the only place `state` returns to
`"healthy"`. Hardware gap defaults are 4-10 s
(`experiment-real/run_experiment.py:526-529`) against 2-8 s in simulation, and
the runner prints `re-engages 4-10s after each fall` at startup. N falls means
N+1 swings.

`--gap-min 1e9 --gap-max 1e9` fakes a single event without a code change, but it
changes the metric: falls-per-bout stops being meaningful, and hardware stops
matching the simulation protocol and the paper.

## Still open: `shift_frac` is not zeroed on heal

`physics.reset()` recentres the payload instantly (hardware `_park()`;
simulation `_settle()` -> `_recenter_payload()`), but the driver still holds
`shift_frac` near 1, so the next `actuate()` tick commands the rod straight back
out and ramps it home over `--shift-ramp-t`. Measured against a stub physics:

```
fall at t = 8.00 s, physics.reset() recentres the payload at t = 8.01 s
   t= 8.00s  frac=1.000
   t= 8.01s  frac=0.990   <-- reset already recentred it
   ...ramps back to 0 over 1.0 s
```

One slow re-swing per recovery, in both drivers (`continual_driver.py:285-303`,
`continual_driver_aif.py:227`). The fix is one line per driver in the fall
branch, `shift_frac = 0.0`, but it is **not applied**: every arm in the published
simulation walked ~1 s under a full payload offset right after each recovery, so
fixing it moves the absolute fall counts (no-adapt 5.18, ours 0.41, grid 1.49,
bo 1.59, esc 1.94) and invalidates the current numbers until they are re-run.
