"""Presence + small-swing test of the CoM mass shifter, robot ON A STAND.

The mass shifter is the rack-and-pinion harness in ``printing/``: a spare servo
drives a slug fore/aft along the trunk, and that is the hardware analogue of the
simulated payload shift. This script answers the two questions you have right
after bolting it on and plugging the servo into a free NyBoard pin:

  1. WHICH PIN is it on, and is anything actually connected there?
  2. Does the slug MOVE when that pin is commanded -- and does the robot feel it?

THE NUMBERS PRINTED ON THE BOARD ARE NOT THE JOINT INDICES
---------------------------------------------------------
Petoi says so themselves (https://bittle.petoi.com/4-connect-the-wires/nyboard):
"The index number of the joint servo has no corresponding relationship with the
PWM PIN on the main board. You should not read the PINs on the PCB board during
assembly." So there is no way to work out which index your servo lands on by
looking at where the lead went -- reading the silkscreen actively misleads. That
is what --scan is for: it sweeps the indices and lets the hardware tell you.

MOTION AT STARTUP IS NOT EVIDENCE
--------------------------------
Connecting REBOOTS the NyBoard -- ardSerial picks the USB modem device precisely
because it restarts the board -- and the firmware then drives every joint,
including whichever one carries the harness, to its rest angle. So the slug
jumping the instant a run starts says nothing about the pin being tested: it is
the boot posture, and it happens on every connect no matter what --port you
passed. Only motion during the creep/swing phases counts, which is why --scan
sweeps all the pins inside ONE connection. If the rest angle drives the slug into
a stop, that is worth fixing at the horn: the servo sits there stalled from every
power-up onward.

WHY THERE IS NO PURELY AUTOMATIC ANSWER
---------------------------------------
The NyBoard drives servos through a PWM expander with no feedback path: an
unplugged pin accepts every command exactly as happily as a plugged one, and the
firmware's joint readback reports what it *commanded*, not what a servo did. So
"is a servo present?" cannot be read off the serial link. What this script does
instead is gather the two signals that do exist:

  * THE OPERATOR. You are standing next to it. The script sweeps and asks.
  * THE IMU. Moving the slug moves the centre of mass, and on a stand that
    usually shows up as a small, repeatable trunk tilt phase-locked to the
    commanded angle. That is measured against a baseline recorded seconds
    earlier with the servo held still, so the noise floor is the robot's own,
    and reported as a correlation with the command rather than a bare peak --
    a number that beats the noise floor AND tracks the command is hard to fake.

  The IMU evidence is corroboration, not proof: a stiff stand, a light slug or a
  clamped chassis can hide a perfectly working shifter. A "no" from the IMU with
  a "yes" from your eyes means the harness works and the trunk simply cannot
  respond -- which is itself worth knowing before you expect a payload shift to
  destabilise anything.

WHAT IT WILL NOT DO
-------------------
* It never commands ports 8-15 (the legs) -- ``--force`` is required to even try,
  because a mistyped pin would otherwise slam a leg against the chassis.
* It swings a few degrees, not the full harness travel. Finding the end stops is
  ``run_experiment.py --mode shift``, which sweeps centred -> shifted; do that
  after this says the servo is alive, and watch for the slug binding.
* The legs are not commanded at all (``--park-legs`` if you want them held at
  the neutral stance), so nothing but the shifter moves.

USAGE
-----
    python shifter_test.py --scan            # which pin is it on?
    python shifter_test.py --port 0          # test that pin (default SHIFT_PORT)
    python shifter_test.py --port 0 --amp 15 --cycles 4
    python shifter_test.py --dry-run --scan  # plumbing rehearsal, no robot

The creep phase steps out one degree at a time with a dwell, so if the rack
binds or the servo buzzes you have time to Ctrl-C -- which returns the servo to
its centre before exiting.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bittle_interface as bi                                   # noqa: E402
import run_experiment as rx                                     # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

LEG_PORTS = set(bi.HIP_PORTS) | set(bi.KNEE_PORTS)
# NyBoard exposes 16 PWM channels; a Bittle drives 8-15 for the legs and 0 for
# the head, leaving 1-7 spare. 0 is included because SHIFT_PORT defaults to it
# (a Bittle without the head servo fitted), but a head servo swinging is NOT the
# mass shifter -- which is exactly what the operator prompt is for.
CANDIDATE_PORTS = [0, 1, 2, 3, 4, 5, 6, 7]
ALL_PORTS = list(range(16))

# The pins are labelled 8-15 on the board in numerical order, so the header that
# looks free next to the ones you can see is not necessarily a spare one: the
# board order and the LEG order are different things. Spell the roles out, so a
# refusal (or a --include-legs scan) says which joint a pin actually drives.
_LEG_NAMES = ["front-left", "front-right", "rear-left", "rear-right"]


def port_role(p):
    """What a pin drives, in words -- '<leg> hip/knee' or 'spare'."""
    for j, q in enumerate(bi.HIP_PORTS):
        if p == q:
            return f"{_LEG_NAMES[j]} hip"
    for j, q in enumerate(bi.KNEE_PORTS):
        if p == q:
            return f"{_LEG_NAMES[j]} knee"
    return "head/spare" if p == 0 else "spare"

# A trunk excursion has to clear both a multiple of the baseline noise and an
# absolute floor, or IMU drift alone would "detect" a shifter on an empty pin.
NOISE_FACTOR = 3.0
ABS_FLOOR_DEG = 0.25
CORR_MIN = 0.5


# ── measurement ──────────────────────────────────────────────────────────────
MIN_TICK = 0.025            # the serial round trip is ~26 ms; this only bites
                            # under --dry-run, where nothing paces the loop


def _pace(tick_start):
    """Hold the sampling loop to MIN_TICK. On hardware this sleeps ~nothing."""
    left = MIN_TICK - (time.perf_counter() - tick_start)
    if left > 0:
        time.sleep(left)


def sample_attitude(link):
    """(roll, pitch) in degrees, or None if the read failed."""
    imu = link.read_imu()
    if imu is None:
        return None
    return np.rad2deg(imu[0]), np.rad2deg(imu[1])


def hold_and_watch(link, seconds):
    """Attitude with nothing commanded: the robot's own noise floor.

    Recorded fresh for every port, seconds before the swing it is compared
    against, so a drifting IMU or a wobbling stand is measured rather than
    assumed away.
    """
    out = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        tick = time.perf_counter()
        rp = sample_attitude(link)
        if rp is not None:
            out.append((tick - t0,) + rp)
        _pace(tick)
    return np.asarray(out) if out else np.zeros((0, 3))


def creep(link, port, center, target, step=1.0, dwell=0.05, verbose=True):
    """Walk the servo to `target` a degree at a time, recording attitude.

    Stepping (rather than jumping) is the whole point: a rack-and-pinion that
    has run out of travel stalls the servo, and a stall you can hear for a
    degree is a stall you can abort. The dwell also lets the slug's inertia
    settle before the next command, so the IMU sees the shift and not the jerk.
    """
    out = []
    t0 = time.perf_counter()
    n = max(1, int(round(abs(target - center) / max(1e-6, step))))
    for i in range(n + 1):
        deg = center + (target - center) * i / n
        tick = time.perf_counter()
        link.joints([port, int(round(deg))])
        time.sleep(dwell)
        rp = sample_attitude(link)
        if rp is not None:
            out.append((tick - t0, deg) + rp)
        if verbose:
            print(f"\r    creep {deg:+6.1f} deg   ", end="", flush=True)
    return out


def swing(link, port, center, amp, cycles, period, verbose=True):
    """Sinusoidal swing about `center`, recording (t, commanded, roll, pitch).

    A sine (rather than a square) is what makes the IMU evidence interpretable:
    the trunk response has to *track* the command, and a correlation against a
    smooth reference separates a real CoM shift from a bump of the bench.
    """
    out = []
    t0 = time.perf_counter()
    total = cycles * period
    while True:
        t = time.perf_counter() - t0
        if t >= total:
            break
        deg = center + amp * np.sin(2.0 * np.pi * t / period)
        link.joints([port, int(round(deg))])
        rp = sample_attitude(link)
        if rp is not None:
            out.append((t, deg) + rp)
        _pace(t0 + t)
        if verbose:
            print(f"\r    swing {t:5.1f}/{total:.0f}s  cmd {deg:+6.1f} deg   ",
                  end="", flush=True)
    return out


# ── verdict ──────────────────────────────────────────────────────────────────
def evidence(base, sw):
    """Did the trunk respond to the commanded swing, beyond its own noise?"""
    out = {"n_base": len(base), "n_swing": len(sw)}
    if len(base) < 5 or len(sw) < 20:
        out["verdict"] = "no data"
        return out
    sw = np.asarray(sw)
    cmd = sw[:, 1]
    for k, col in (("roll", 2), ("pitch", 3)):
        b_pp = float(np.ptp(base[:, col - 1]))
        s = sw[:, col]
        s_pp = float(np.ptp(s))
        # Correlation is undefined for a flat channel; a frozen IMU lands here.
        if np.std(s) < 1e-9 or np.std(cmd) < 1e-9:
            corr = 0.0
        else:
            corr = float(np.corrcoef(cmd, s)[0, 1])
        out[k] = dict(base_pp=b_pp, swing_pp=s_pp, corr=corr,
                      moved=bool(s_pp > max(NOISE_FACTOR * b_pp, ABS_FLOOR_DEG)
                                 and abs(corr) > CORR_MIN))
    out["moved"] = bool(out["roll"]["moved"] or out["pitch"]["moved"])
    return out


def print_evidence(ev, indent="    "):
    if ev.get("verdict") == "no data":
        print(f"{indent}!! not enough IMU samples to judge")
        return
    print(f"{indent}axis    noise floor   swing        corr with command")
    for k in ("roll", "pitch"):
        v = ev[k]
        mark = "  <-- tracks the command" if v["moved"] else ""
        print(f"{indent}{k:6s} {v['base_pp']:7.2f} deg  {v['swing_pp']:7.2f} deg"
              f"  {v['corr']:+6.2f}{mark}")


def ask(prompt, default=None, no_prompt=False):
    """y/n from the operator; `default` when there is nobody to ask."""
    if no_prompt:
        return default
    try:
        while True:
            r = input(prompt).strip().lower()
            if r in ("y", "yes"):
                return True
            if r in ("n", "no"):
                return False
            if r == "" and default is not None:
                return default
    except EOFError:
        return default


# ── one port ─────────────────────────────────────────────────────────────────
def test_port(a, link, port, quick=False, verbose=True):
    """Baseline -> creep out and back -> swing -> verdict, for one pin.

    Judge only what moves during the creep and swing phases: connecting rebooted
    the board and every joint snapped to its rest angle before any of this ran.
    """
    amp = a.amp if not quick else min(a.amp, 10.0)
    cycles = a.cycles if not quick else 1
    print(f"\n  -- port {port} ({port_role(port)}): centre {a.center:+g} deg, "
          f"swinging +-{amp:g} deg --")
    print(f"    holding still for {a.settle:g}s (noise floor)")
    link.joints([port, int(round(a.center))])
    time.sleep(0.4)
    base = hold_and_watch(link, a.settle)

    crp = []
    crp.append(creep(link, port, a.center, a.center + amp, a.step, a.dwell,
                     verbose))
    crp.append(creep(link, port, a.center + amp, a.center - amp, a.step, a.dwell,
                     verbose))
    crp.append(creep(link, port, a.center - amp, a.center, a.step, a.dwell,
                     verbose))
    if verbose:
        print()
    sw = swing(link, port, a.center, amp, cycles, a.period, verbose)
    if verbose:
        print()
    link.joints([port, int(round(a.center))])          # always leave it centred

    ev = evidence(base, sw)
    print_evidence(ev)
    seen = ask("    did the slug visibly move? [y/n] ", default=None,
               no_prompt=a.no_prompt)
    ev["operator"] = seen
    ev["port"] = port
    ev["creep"] = [list(r) for leg in crp for r in leg]
    ev["swing"] = [list(r) for r in sw]
    ev["baseline"] = base.tolist()
    return ev


def mode_scan(a, link):
    """Sweep each spare pin in turn until one of them moves the slug."""
    if a.ports:
        ports = list(a.ports)
    elif a.include_legs:
        ports = list(ALL_PORTS)
    else:
        ports = [p for p in CANDIDATE_PORTS if p not in LEG_PORTS]
    print(f"\n  SCAN over ports {ports}. Each is swung +-{min(a.amp, 10):g} deg "
          f"in turn;\n  watch the slug, not the legs.\n  A head servo (usually "
          f"port 0) will also swing: that is not the shifter.")
    legs = [p for p in ports if p in LEG_PORTS]
    if legs:
        print("  !! this sweep includes LEG pins -- robot on a stand, legs free "
              "to move:\n" + "\n".join(f"     {p:2d}  {port_role(p)}"
                                        for p in legs))
    found, results = [], []
    for port in ports:
        ev = test_port(a, link, port, quick=True, verbose=not a.quiet)
        results.append(ev)
        if ev.get("operator") is True:
            found.append(port)
            print(f"\n  ** the mass shifter is on port {port} **")
            if not ask("    keep scanning the remaining ports? [y/n] ",
                       default=False, no_prompt=a.no_prompt):
                break
        elif ev.get("operator") is None and ev.get("moved"):
            found.append(port)
            print(f"    (no operator answer, but the trunk tracked the command "
                  f"on port {port})")
    print("\n  == scan result ==")
    if not found:
        moved = [r["port"] for r in results if r.get("moved")]
        print("  no port moved the slug." + (
            f" The IMU did respond on {moved} -- worth a closer look with "
            f"--port." if moved else
            "\n  Check: is the servo lead on the signal/GND/V pins the right way "
            "round, is the battery on (the PWM rail is unpowered over USB alone "
            "on some boards), and does the pinion actually engage the rack?"))
    else:
        p = found[0]
        print(f"  mass shifter on port {p} ({port_role(p)}).")
        if p in LEG_PORTS:
            print(f"  !! that pin is the {port_role(p)} servo's. Both cannot "
                  f"live there: every control tick commands it as a leg joint, "
                  f"so\n     whichever command lands last in the packet wins "
                  f"and the gait is broken.\n     Move the shifter lead to a "
                  f"spare pin (1-7) and re-run --scan before any walking run.")
            return results
        print(f"  Carry it into every run as --shift-port {p}")
        if p != bi.SHIFT_PORT:
            print(f"  (bittle_interface.SHIFT_PORT is {bi.SHIFT_PORT}; set it to "
                  f"{p} to make that the default)")
        print(f"  Next: python run_experiment.py --mode shift --shift-port {p}\n"
              f"        finds the end stops -- watch for the slug binding, then "
              f"set --shift-centered / --shift-shifted.")
    return results


# ── wiggle ───────────────────────────────────────────────────────────────────
def mode_wiggle(a, link):
    """Interactive index prober: type an index, watch it move, repeat.

    This is the swap-testing tool. Working out whether a servo is dead or a
    header is empty means moving one lead and commanding a different index, over
    and over -- and reconnecting between attempts would reboot the board and snap
    every joint to its rest angle, motion you would then have to discount. One
    connection stays open here, so everything that moves, moved because you asked
    for it.
    """
    print(f"\n  WIGGLE on one connection (no reboots, so every motion counts).\n"
          f"  Type an index 0-15 to swing it +-{a.amp:g} deg about its rest "
          f"angle; blank to finish.\n  Legs are 8-15 -- robot on a stand if you "
          f"are going to move those.")
    results = []
    while True:
        try:
            r = input("\n    index? ").strip()
        except EOFError:
            break
        if not r:
            break
        try:
            idx = int(r)
        except ValueError:
            print("    an index, 0-15")
            continue
        if not 0 <= idx <= 15:
            print("    0-15 only")
            continue
        role = port_role(idx)
        if idx in LEG_PORTS and not a.force and not ask(
                f"    index {idx} is the {role} -- move it anyway? [y/n] ",
                default=False, no_prompt=a.no_prompt):
            continue
        if idx in set(bi.HIP_PORTS):
            base = bi.HIP_OFFSET_DEG
        elif idx in set(bi.KNEE_PORTS):
            base = bi.KNEE_OFFSET_DEG
        else:
            base = a.center
        n = max(1, a.cycles)
        print(f"    index {idx} ({role}): {n} wiggle{'s' if n > 1 else ''} of "
              f"+-{a.amp:g} deg about {base:g} -- watch and listen")
        rows, t0 = [], time.perf_counter()
        for _ in range(n):
            for d in (+a.amp, 0.0, -a.amp, 0.0):
                link.joints([idx, int(round(base + d))])
                time.sleep(0.25)
                rp = sample_attitude(link)
                if rp is not None:
                    rows.append((time.perf_counter() - t0, base + d) + rp)
        link.joints([idx, int(round(base))])
        seen = ask("    did it move? [y/n] ", default=None,
                   no_prompt=a.no_prompt)
        print("    -> " + ({True: f"index {idx} drives a working servo",
                            False: f"nothing on index {idx}"}.get(
                                seen, "no answer recorded")))
        results.append(dict(port=idx, creep=[], swing=rows, baseline=[],
                            moved=None, operator=seen))
    return results


# ── travel ───────────────────────────────────────────────────────────────────
def mode_travel(a, link, port):
    """Find the harness end stops by creeping outward and asking, one side at a
    time.

    ``--center`` is NOT the middle of the rack -- the servo's calibrated zero
    lands wherever the pinion happened to be when the horn went on, so one
    direction can be against a stop from the first degree. Nothing on the link
    reports a stall, so the operator is the limit switch: this creeps a few
    degrees, asks, and treats the last confirmed angle as the end.
    """
    print(f"\n  TRAVEL on port {port} ({port_role(port)}), from "
          f"{a.center:+g} deg, at most +-{a.max_travel:g} deg.\n"
          f"  It creeps {a.ask_every:g} deg at a time and asks. Answer n as soon "
          f"as the slug\n  reaches the end of the rack -- BEFORE it presses on "
          f"the stop, not after.")
    link.joints([port, int(round(a.center))])
    time.sleep(0.5)

    rows, t_off, limits = [], 0.0, {}
    for sign, name in ((+1.0, "plus"), (-1.0, "minus")):
        print(f"\n  -- {name} direction --")
        deg = last_ok = float(a.center)
        while abs(deg - a.center) < a.max_travel - 1e-9:
            span = min(a.ask_every, a.max_travel - abs(deg - a.center))
            seg = creep(link, port, deg, deg + sign * span, a.step, a.dwell,
                        verbose=not a.quiet)
            rows += [(t_off + r[0],) + tuple(r[1:]) for r in seg]
            t_off += seg[-1][0] if seg else 0.0
            deg += sign * span
            if not a.quiet:
                print()
            if not ask(f"    at {deg:+.0f} deg -- room for more? [y/n] ",
                       default=True, no_prompt=a.no_prompt):
                break
            last_ok = deg
        limits[name] = last_ok
        seg = creep(link, port, deg, a.center, a.step, a.dwell,
                    verbose=not a.quiet)          # back off before the other side
        rows += [(t_off + r[0],) + tuple(r[1:]) for r in seg]
        t_off += seg[-1][0] if seg else 0.0
        if not a.quiet:
            print()

    lo, hi = limits["minus"], limits["plus"]
    m = a.margin
    lo_s, hi_s = lo + m if lo < a.center else lo, hi - m if hi > a.center else hi
    print(f"\n  == travel on port {port} ==")
    print(f"  you called the stops at {lo:+.0f} .. {hi:+.0f} deg "
          f"({hi - lo:.0f} deg of usable rack)")
    if hi - lo < 2 * a.margin + 1:
        print("  !! that is barely any travel. Re-seat the servo horn so the "
              "zero sits at one end of the rack rather than against a stop.")
        return [dict(port=port, creep=rows, swing=[], baseline=[],
                     moved=None, operator=None, travel=[lo, hi])]
    print(f"  with {m:g} deg of margin, drive it between {lo_s:+.0f} and "
          f"{hi_s:+.0f} deg:")
    print(f"    slug centred at {lo_s:+.0f}:  --shift-port {port} "
          f"--shift-centered {lo_s:.0f} --shift-shifted {hi_s:.0f}")
    print(f"    slug centred at {hi_s:+.0f}:  --shift-port {port} "
          f"--shift-centered {hi_s:.0f} --shift-shifted {lo_s:.0f}")
    print("  Pick by where the slug sits: centred = over the nominal CoM, "
          "shifted = the\n  displaced payload the experiment ramps to. Then "
          "`run_experiment.py --mode shift`\n  with those flags cycles the full "
          "ramp -- watch that neither end binds.")
    return [dict(port=port, creep=rows, swing=[], baseline=[],
                 moved=None, operator=None, travel=[lo, hi])]


# ── main ─────────────────────────────────────────────────────────────────────
def save(a, results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(RESULTS_DIR, f"shiftertest_{stamp}.npz")
    meta = dict(center=a.center, amp=a.amp, cycles=a.cycles, period=a.period,
                step=a.step, dwell=a.dwell, settle=a.settle, dry_run=a.dry_run,
                ports=[r["port"] for r in results],
                moved=[bool(r.get("moved")) for r in results],
                operator=[r.get("operator") for r in results],
                travel=[r.get("travel") for r in results])
    arrays = {}
    for r in results:
        # (t, commanded_deg, roll_deg, pitch_deg) for the two phases; the
        # baseline has no command column: (t, roll_deg, pitch_deg).
        for key in ("creep", "swing", "baseline"):
            arrays[f"{key}_{r['port']}"] = np.asarray(r.pop(key), dtype=float)
    np.savez_compressed(path, meta=json.dumps(meta, default=float), **arrays)
    print(f"\n  saved {path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wiggle", action="store_true",
                    help="interactive prober: type an index, it moves, repeat -- "
                         "all on ONE connection, so no boot snaps to discount. "
                         "This is the tool for swap-testing a suspect servo")
    ap.add_argument("--find-travel", action="store_true",
                    help="creep outward one side at a time, asking where the "
                         "rack ends, and print the --shift-centered/"
                         "--shift-shifted pair. Do this once the swing test has "
                         "found the pin: the servo zero is NOT mid-travel")
    ap.add_argument("--scan", action="store_true",
                    help="try every spare pin in turn (do this if you do not "
                         "know which one you plugged into)")
    ap.add_argument("--port", type=int, default=bi.SHIFT_PORT,
                    help=f"pin to test (default SHIFT_PORT={bi.SHIFT_PORT})")
    ap.add_argument("--ports", type=int, nargs="+", default=None,
                    help="explicit candidate list for --scan")
    ap.add_argument("--force", action="store_true",
                    help="allow a LEG port (8-15) to be commanded -- a mistyped "
                         "pin folds a leg into the chassis, so this is opt-in")
    ap.add_argument("--include-legs", action="store_true",
                    help="--scan sweeps all 16 pins, legs included. Use this if "
                         "the shifter is plugged into a pin in the 8-15 block; "
                         "robot on a stand, legs free to swing")
    ap.add_argument("--park-legs", action="store_true",
                    help="hold the legs at the neutral stance during the test "
                         "(off by default: nothing but the shifter should move)")

    g = ap.add_argument_group("swing")
    g.add_argument("--center", type=float, default=0.0, help="[deg]")
    g.add_argument("--amp", type=float, default=10.0,
                   help="[deg] swing amplitude about --center. A few degrees is "
                        "the point: the end stops are run_experiment --mode shift")
    g.add_argument("--cycles", type=int, default=3)
    g.add_argument("--period", type=float, default=2.0,
                   help="[s] per full back-and-forth cycle")
    g.add_argument("--step", type=float, default=1.0,
                   help="[deg] creep step -- 1 deg is the servo command quantum")
    g.add_argument("--dwell", type=float, default=0.05,
                   help="[s] held at each creep step, so a stall is audible "
                        "before the next one")
    g.add_argument("--ask-every", type=float, default=5.0,
                   help="[deg] travelled between end-stop questions "
                        "(--find-travel)")
    g.add_argument("--max-travel", type=float, default=45.0,
                   help="[deg] hard cap either side of --center (--find-travel)")
    g.add_argument("--margin", type=float, default=2.0,
                   help="[deg] kept clear of each stop in the recommendation")
    g.add_argument("--settle", type=float, default=1.5,
                   help="[s] of held-still attitude recorded as the noise floor")

    g = ap.add_argument_group("robot")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--no-prompt", action="store_true",
                   help="do not ask the operator; judge on the IMU alone (which "
                        "cannot tell a shifter from any other moving mass)")
    g.add_argument("--quiet", action="store_true", help="no live progress line")
    rx.add_wifi_args(g)
    g.add_argument("--imu-units", choices=["auto", "deg", "rad"], default="auto")
    g.add_argument("--keep-gyro", action="store_true",
                   help="do NOT deactivate the firmware's gyro balancing")
    g.add_argument("--roll-sign", type=float, default=1.0)
    g.add_argument("--pitch-sign", type=float, default=1.0)
    a = ap.parse_args()

    if a.scan:
        req = a.ports or (ALL_PORTS if a.include_legs else
                          [p for p in CANDIDATE_PORTS if p not in LEG_PORTS])
    else:
        req = [a.port]
    bad = [p for p in req if p in LEG_PORTS]
    if bad and not (a.force or (a.scan and a.include_legs)):
        raise SystemExit(
            "refusing to command "
            + ", ".join(f"port {p} ({port_role(p)})" for p in bad)
            + ".\n8-15 are the gait's own joints, so this script leaves them "
              "alone unless you say otherwise -- a mistyped pin folds a leg "
              "into the chassis.\nIf the shifter really is on one of them, put "
              "the robot on a stand and pass --force (or --scan --include-legs)."
              "\nBe aware that pin is then double-booked: bittle_interface "
              "drives it as a leg joint every control tick, so the shifter has "
              "to move to a spare pin (1-7) before any walking run.")

    link = rx.make_link(a)
    link.connect()
    try:
        if a.park_legs:
            link.joints(bi.BittleLink.neutral_pose())
            time.sleep(1.0)
        if a.wiggle:
            results = mode_wiggle(a, link)
        elif a.find_travel:
            results = mode_travel(a, link, a.port)
        elif a.scan:
            results = mode_scan(a, link)
        else:
            results = [test_port(a, link, a.port, verbose=not a.quiet)]
        if not (a.scan or a.find_travel or a.wiggle):
            ev = results[0]
            print()
            if ev.get("operator") is True:
                print(f"  ok: the mass shifter on port {a.port} moves."
                      + ("" if ev.get("moved") else
                         "\n  The IMU did not see it, which is fine on a rigid "
                         "stand -- but it also means a payload shift will not "
                         "show up in the tilt CUSUM until the robot is on the "
                         "floor and free to lean."))
            elif ev.get("operator") is False:
                print(f"  !! nothing moved on port {a.port}. Try --scan; check "
                      f"the lead orientation and that the servo rail is powered "
                      f"(battery on, not USB alone).")
            else:
                print(f"  the trunk {'DID' if ev.get('moved') else 'did NOT'} "
                      f"track the commanded swing on port {a.port}"
                      + ("." if ev.get("moved") else
                         " -- but with no operator answer that is not conclusive: "
                         "a stiff stand hides a working shifter."))
            print(f"  Next: python run_experiment.py --mode shift --shift-port "
                  f"{a.port}   (end stops)")
        if results:
            save(a, results)
    except KeyboardInterrupt:
        print("\n  aborted -- centring the servo")
        try:
            link.joints([a.port, int(round(a.center))])
        except Exception:
            pass
    finally:
        link.close()


if __name__ == "__main__":
    main()
