"""Pluggable ground terrain for the CPG locomotion experiments.

The episode code senses foot contact with ``getContactPoints(bodyA=0,
linkIndexA=-1, ...)``, i.e. it assumes the ground is body 0 and that the
walking surface is that body's *base* link. To keep that contract, the sloped
terrain is built as a single heightfield body (one base collision shape) rather
than as multiple boxes/links — so foot-contact sensing keeps working unchanged.

Selection is via the module-level ``TERRAIN_CONFIG``; an experiment sets it
before running, e.g.::

    from methods import terrain
    terrain.TERRAIN_CONFIG = {"kind": "sloped", "slope_deg": 10.0,
                              "slope_start_y": 2.0}

The default is flat (plain ``plane.urdf``), so existing experiments are
unaffected.

The robot walks along +Y (forward). A "sloped" terrain is flat for
``y < slope_start_y`` and then ramps upward at ``slope_deg`` degrees, so the
robot meets the incline part-way through the task.
"""

import numpy as np

# Mutable global selected by the experiment driver before the env is built.
TERRAIN_CONFIG = {"kind": "flat"}

# Id of the ground body, recorded by build_ground so apply_dynamic_friction can
# address it without threading it through the episode code.
_ground_id = None

# Named surface friction coefficients (lateral friction of the ground plane).
# "ice" lowered to 0.15 so a friction *drop* onto it causes a real slip (the
# transition-recovery test needs an actual disturbance to recover from). This
# trades some falls for a measurable adaptation signal.
SURFACES = {"ice": 0.15, "slick": 0.45, "normal": 0.7, "grip": 1.1, "rubber": 1.6}


def build_ground(p):
    """Create the ground body and return its id. Must be called first so the
    ground is body 0."""
    global _ground_id
    kind = TERRAIN_CONFIG.get("kind", "flat")
    if kind == "flat":
        _ground_id = p.loadURDF("plane.urdf")
        return _ground_id
    if kind == "friction":
        # Flat plane (body 0); friction is set per-step from the robot's forward
        # position by apply_dynamic_friction. Start at the base friction.
        _ground_id = p.loadURDF("plane.urdf")
        p.changeDynamics(_ground_id, -1,
                         lateralFriction=float(TERRAIN_CONFIG.get("base_mu", 0.7)))
        return _ground_id
    if kind == "sloped":
        # Single ramp: flat until slope_start_y, then up at slope_deg.
        segs = [(float(TERRAIN_CONFIG.get("slope_start_y", 2.0)),
                 float(TERRAIN_CONFIG.get("slope_deg", 10.0)))]
        _ground_id = _build_heightfield(p, _piecewise_height_fn(segs))
        return _ground_id
    if kind == "multislope":
        # Piecewise terrain: list of (y_start, slope_deg) segments. Slope is 0
        # before the first y_start and equal to the last preceding segment after.
        segs = TERRAIN_CONFIG["segments"]
        _ground_id = _build_heightfield(p, _piecewise_height_fn(segs))
        return _ground_id
    raise ValueError(f"Unknown terrain kind: {kind!r}")


def sample_friction(seed, ice_start=(1.6, 2.4), ice_len=(1.5, 2.0),
                    n_extra=(1, 2), zone_len=(0.8, 1.6), base_mu=0.7,
                    palette=("slick", "normal", "grip", "rubber")):
    """Sample a friction terrain spec that ALWAYS contains one reachable
    normal->ice drop.

    The robot starts on `base_mu` (normal, 0.7) and crosses onto a guaranteed
    ice patch (0.15) at ice_start — a clean, reachable friction drop of ~0.55 to
    slip on and recover from. The ice patch is long enough (ice_len) that the
    post-transition analysis window stays on ice. A few extra random zones
    (from `palette`, excluding ice) follow for variety. Terrain stays flat — only
    friction varies. RNG decoupled from optimizer seeds via a +7000 offset.
    """
    rng = np.random.default_rng(7000 + int(seed))
    y = float(rng.uniform(*ice_start))
    zones = [(round(y, 3), SURFACES["ice"], "ice")]   # guaranteed normal->ice drop
    y += float(rng.uniform(*ice_len))
    for _ in range(int(rng.integers(n_extra[0], n_extra[1] + 1))):
        surf = palette[int(rng.integers(0, len(palette)))]
        zones.append((round(y, 3), SURFACES[surf], surf))
        y += float(rng.uniform(*zone_len))
    return {"kind": "friction", "zones": zones, "base_mu": base_mu,
            "n_zones": len(zones), "ice_drop_y": round(zones[0][0], 3)}


def friction_at(y):
    """Lateral friction coefficient at forward position y for the current
    friction TERRAIN_CONFIG (base_mu before the first zone)."""
    mu = float(TERRAIN_CONFIG.get("base_mu", 0.7))
    for z in TERRAIN_CONFIG.get("zones", []):
        if y >= z[0]:
            mu = z[1]
        else:
            break
    return mu


def apply_dynamic_friction(p, robot_id, pos_y):
    """If the terrain is a friction terrain, set the ground's lateral friction to
    the value of the zone the robot is currently in. No-op otherwise. Called
    once per simulation step from the episode loops with the robot's forward
    position (pos_y)."""
    if TERRAIN_CONFIG.get("kind") != "friction" or _ground_id is None:
        return
    p.changeDynamics(_ground_id, -1, lateralFriction=friction_at(pos_y))


def sample_multislope(seed, n_range=(1, 4),
                      first_start=(1.5, 2.5),
                      seg_len=(1.2, 2.2),
                      slope_abs_deg=(4.0, 10.0)):
    """Sample a multi-slope terrain spec for a given seed.

    Returns a config dict with a random number of slope segments (uniform in
    n_range), each with a random length, and a random slope angle of random
    sign (|angle| in slope_abs_deg). A flat segment is appended after the last
    slope. Decoupled from optimizer RNGs via a +9000 offset on the seed.
    """
    rng = np.random.default_rng(9000 + int(seed))
    n_slopes = int(rng.integers(n_range[0], n_range[1] + 1))
    y = float(rng.uniform(*first_start))
    segments = []
    for _ in range(n_slopes):
        slope = float(rng.uniform(*slope_abs_deg)) * (1 if rng.random() < 0.5 else -1)
        segments.append((round(y, 3), round(slope, 2)))
        y += float(rng.uniform(*seg_len))
    segments.append((round(y, 3), 0.0))   # flat tail after the last slope
    return {"kind": "multislope", "segments": segments, "n_slopes": n_slopes}


def _piecewise_height_fn(segments):
    """Return h(y) for a piecewise-constant-slope profile defined by sorted
    (y_start, slope_deg) segments. Slope is 0 before the first y_start. The
    profile is continuous (piecewise-linear) and 0 on the flat start region."""
    seg = sorted(segments, key=lambda s: s[0])
    ys = np.array([s[0] for s in seg], dtype=float)
    tans = np.array([np.tan(np.deg2rad(s[1])) for s in seg], dtype=float)

    def fn(y):
        y = np.asarray(y, dtype=float)
        order = np.argsort(y)
        ysrt = y[order]
        h = np.zeros_like(ysrt)
        for k in range(1, len(ysrt)):
            s_idx = np.searchsorted(ys, ysrt[k - 1], side="right") - 1
            slope = tans[s_idx] if s_idx >= 0 else 0.0
            h[k] = h[k - 1] + slope * (ysrt[k] - ysrt[k - 1])
        out = np.empty_like(h)
        out[order] = h
        return out

    return fn


def _build_heightfield(p, height_fn, mesh_dx=0.05, mesh_dy=0.05,
                       n_rows=256, n_cols=512):
    """Build a heightfield body (id 0) from a height-of-y function, calibrated
    so the flat start region sits at z=0 (matching plane.urdf)."""
    # World Y of each column (terrain centred on its base position).
    j = np.arange(n_cols)
    col_y = (j - (n_cols - 1) / 2.0) * mesh_dy
    col_h = np.asarray(height_fn(col_y), dtype=float)

    # heightfieldData is row-major: index = row + col * n_rows. Height is
    # constant across rows (X), varying across columns (Y).
    data = np.repeat(col_h[None, :], n_rows, axis=0)          # (n_rows, n_cols)
    height_list = data.flatten(order="F").astype(np.float64)  # row + col*n_rows

    shape = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        meshScale=[mesh_dx, mesh_dy, 1.0],
        heightfieldData=height_list.tolist(),
        numHeightfieldRows=n_rows,
        numHeightfieldColumns=n_cols,
    )
    terrain = p.createMultiBody(0, shape)
    p.changeVisualShape(terrain, -1, rgbaColor=[0.55, 0.55, 0.6, 1.0])

    # Cast a ray down at the start (0,0) and shift the body so the flat region
    # is exactly at z=0.
    p.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0, 1])
    surf0 = _surface_z(p, 0.0, 0.0)
    if surf0 is not None:
        p.resetBasePositionAndOrientation(terrain, [0, 0, -surf0], [0, 0, 0, 1])
    return terrain


def _surface_z(p, x, y, z_top=5.0, z_bot=-5.0):
    hit = p.rayTest([x, y, z_top], [x, y, z_bot])[0]
    if hit[0] < 0:
        return None
    return hit[3][2]   # world z of hit point


if __name__ == "__main__":
    # Self-test: single ramp at z=0 on the flat part, plus a few sampled
    # multi-slope terrains (check continuity and that the flat start is at z=0).
    import pybullet as pb
    import pybullet_data

    pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0, 0, -9.8)

    print("== single sloped ==")
    TERRAIN_CONFIG = {"kind": "sloped", "slope_deg": 10.0, "slope_start_y": 2.0}
    build_ground(pb)
    tan_s = np.tan(np.deg2rad(10.0))
    for y in [0, 1, 2, 4, 6]:
        z = _surface_z(pb, 0.0, float(y))
        print(f"  y={y}: surf_z={z:.3f} expected={max(0.0,(y-2.0))*tan_s:.3f}")

    for seed in range(5):
        pb.resetSimulation()
        cfg = sample_multislope(seed)
        print(f"\n== multislope seed={seed}: n_slopes={cfg['n_slopes']} "
              f"segments={cfg['segments']} ==")
        TERRAIN_CONFIG = cfg
        build_ground(pb)
        zs = [(_surface_z(pb, 0.0, float(y))) for y in range(0, 8)]
        print("  surf_z y=0..7:", [f"{z:.2f}" for z in zs])
    pb.disconnect()
