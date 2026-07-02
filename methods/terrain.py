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
    if kind == "natural":
        # Natural landscape: forward bands of grass/gravel/rocks/river, each with
        # its own surface geometry (2-D heightfield) and friction (via zones).
        grid, dx, dy, base_y = _natural_height_grid(TERRAIN_CONFIG)
        _ground_id = _build_heightfield(p, None, mesh_dx=dx, mesh_dy=dy,
                                        height_grid=grid, base_xy=(0.0, base_y),
                                        rgba=(0.45, 0.43, 0.38, 1.0))
        return _ground_id
    if kind == "obstacles":
        # Isaac-Gym-style discrete obstacles: flat ground with scattered raised
        # rectangular platforms the robot must step over (uniform friction).
        grid, dx, dy, base_y = _obstacles_height_grid(TERRAIN_CONFIG)
        _ground_id = _build_heightfield(p, None, mesh_dx=dx, mesh_dy=dy,
                                        height_grid=grid, base_xy=(0.0, base_y),
                                        rgba=(0.5, 0.5, 0.55, 1.0))
        p.changeDynamics(_ground_id, -1,
                         lateralFriction=float(TERRAIN_CONFIG.get("base_mu", 0.7)))
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


def sample_friction_long(seed, reach=60.0, first_start=(1.5, 2.5),
                         zone_len=(1.5, 3.5), base_mu=0.7,
                         palette=("ice", "slick", "normal", "grip", "rubber")):
    """Friction terrain for a long continuous run: a sequence of random friction
    zones tiling the path from ~first_start out to `reach` metres, so the robot
    keeps crossing surface changes as it walks forward. RNG seeded 7000+seed
    (same family as sample_friction)."""
    rng = np.random.default_rng(7000 + int(seed))
    y = float(rng.uniform(*first_start))
    zones = []
    while y < reach:
        surf = palette[int(rng.integers(0, len(palette)))]
        zones.append((round(y, 3), SURFACES[surf], surf))
        y += float(rng.uniform(*zone_len))
    return {"kind": "friction", "zones": zones, "base_mu": base_mu,
            "n_zones": len(zones)}


# Natural-landscape surfaces: name -> (lateral friction, geometry, amplitude[m]).
# Geometry is kept traversable for the Laikago (foot clearance ~0.05-0.08 m):
# the difficulty is the friction *contrast* between bands, not impassable steps.
NATURAL_SURFACES = {
    "grass":  (0.70, "noise", 0.010),   # near-flat, gentle undulation
    "gravel": (0.55, "noise", 0.022),   # small high-frequency roughness
    "rocks":  (0.95, "bumps", 0.05),    # low scattered rocks to step over
    "river":  (0.20, "dip",   0.07),    # wet channel: a slippery depression
}
# muted RGB colours for top-down visualisation
NATURAL_COLORS = {"grass": (0.30, 0.55, 0.25), "gravel": (0.55, 0.52, 0.48),
                  "rocks": (0.40, 0.38, 0.36), "river": (0.20, 0.45, 0.75)}


def sample_natural(seed, reach=20.0, band_len=(2.5, 4.5), start_grass=2.5):
    """Sample a natural-landscape transect: forward bands of grass / gravel /
    rocks / river (random order and lengths), starting on grass. Returns the
    band layout and the friction zones (for friction_at). RNG seeded 5000+seed."""
    rng = np.random.default_rng(5000 + int(seed))
    names = ["gravel", "rocks", "river", "grass"]
    bands = [(0.0, "grass")]                       # start region
    y = float(start_grass)
    while y < reach:
        nm = names[int(rng.integers(0, len(names)))]
        if nm == bands[-1][1]:                     # avoid identical back-to-back
            nm = "grass" if nm != "grass" else "gravel"
        bands.append((round(y, 3), nm))
        y += float(rng.uniform(*band_len))
    zones = [(yb, NATURAL_SURFACES[nm][0], nm) for (yb, nm) in bands]
    return {"kind": "natural", "bands": bands, "zones": zones,
            "base_mu": NATURAL_SURFACES["grass"][0], "reach": reach, "seed": seed}


def _band_at(bands, y):
    nm = bands[0][1]
    for (yb, b) in bands:
        if y >= yb:
            nm = b
        else:
            break
    return nm


def _natural_height_grid(cfg, mesh=0.06, x_half=3.0):
    """2-D height grid (rows=X lateral, cols=Y forward) for a natural transect:
    grass gentle undulation, gravel fine roughness, rocks scattered bumps, river
    a smooth slippery channel. Returns (grid, dx, dy, base_y)."""
    from scipy.ndimage import gaussian_filter
    bands = cfg["bands"]; reach = float(cfg.get("reach", 20.0))
    rng = np.random.default_rng(6000 + int(cfg.get("seed", 0)))
    dx = dy = float(mesh)
    base_y = reach / 2.0                           # centre the grid on the path
    half_y = reach / 2.0 + 3.0
    n_cols = int(2 * half_y / dy)
    n_rows = int(2 * x_half / dx)
    yw = base_y + (np.arange(n_cols) - (n_cols - 1) / 2.0) * dy   # forward
    xw = (np.arange(n_rows) - (n_rows - 1) / 2.0) * dx            # lateral
    band_of_col = np.array([_band_at(bands, y) for y in yw])

    grid = np.zeros((n_rows, n_cols))
    fine = gaussian_filter(rng.standard_normal((n_rows, n_cols)), sigma=1.2)
    fine /= (np.abs(fine).max() + 1e-9)
    gentle = gaussian_filter(rng.standard_normal((n_rows, n_cols)), sigma=4.0)
    gentle /= (np.abs(gentle).max() + 1e-9)
    for j, nm in enumerate(band_of_col):
        amp = NATURAL_SURFACES[nm][2]
        if NATURAL_SURFACES[nm][1] == "noise":
            grid[:, j] = amp * (fine[:, j] if nm == "gravel" else gentle[:, j])

    rock_cols = np.where(band_of_col == "rocks")[0]
    if rock_cols.size:
        y_lo, y_hi = yw[rock_cols.min()], yw[rock_cols.max()]
        XX, YY = np.meshgrid(xw, yw, indexing="ij")
        for _ in range(max(4, int(2.0 * (y_hi - y_lo)))):
            cx = rng.uniform(-x_half * 0.8, x_half * 0.8)
            cy = rng.uniform(y_lo, y_hi)
            w = rng.uniform(0.12, 0.30)
            a = NATURAL_SURFACES["rocks"][2] * rng.uniform(0.5, 1.2)
            grid += a * np.exp(-(((XX - cx) ** 2 + (YY - cy) ** 2) / (2 * w ** 2)))

    for i, (yb, nm) in enumerate(bands):
        if nm != "river":
            continue
        y_end = bands[i + 1][0] if i + 1 < len(bands) else reach
        L = max(y_end - yb, 0.5)
        sel = (yw >= yb) & (yw < y_end)
        prof = -NATURAL_SURFACES["river"][2] * 0.5 * (1 - np.cos(2 * np.pi * (yw[sel] - yb) / L))
        grid[:, sel] = prof[None, :]              # valley across full width
    return grid, dx, dy, base_y


def sample_obstacles(seed, reach=20.0, first_start=2.5, density=0.6,
                     height=(0.04, 0.08), size=(0.25, 0.6), base_mu=0.7):
    """Isaac-Gym-style discrete-obstacle field: scattered raised rectangular
    platforms on otherwise flat ground, from ~first_start out to `reach` metres,
    that the robot has to step over (uniform friction `base_mu`). `density` is
    roughly obstacles per metre of forward travel; heights kept traversable for
    the Laikago. RNG seeded 8000+seed (decoupled from optimizer seeds)."""
    rng = np.random.default_rng(8000 + int(seed))
    n = max(0, int(round(density * (reach - first_start))))   # 0 => flat ground
    obs = []
    for _ in range(n):
        cy = float(rng.uniform(first_start, reach))
        cx = float(rng.uniform(-1.6, 1.6))
        hx = float(rng.uniform(*size)) / 2.0
        hy = float(rng.uniform(*size)) / 2.0
        h = float(rng.uniform(*height))
        obs.append((cx, cy, hx, hy, h))
    return {"kind": "obstacles", "obstacles": obs, "reach": float(reach),
            "base_mu": float(base_mu), "seed": int(seed), "n_obs": len(obs)}


def _obstacles_height_grid(cfg, mesh=0.05, x_half=3.0):
    """2-D height grid (rows=X lateral, cols=Y forward) for a discrete-obstacle
    field: flat ground with scattered raised mounds. The mounds are smooth
    (Gaussian) rather than hard boxes — an open-loop CPG gait can ride over a
    smooth rise but a vertical box edge is a wall that topples it. Each obstacle
    (cx, cy, hx, hy, h) becomes a Gaussian bump of peak height h and footprint
    ~ (hx, hy). Returns (grid, dx, dy, base_y)."""
    reach = float(cfg.get("reach", 20.0))
    dx = dy = float(mesh)
    base_y = reach / 2.0
    half_y = reach / 2.0 + 3.0
    n_cols = int(2 * half_y / dy)
    n_rows = int(2 * x_half / dx)
    yw = base_y + (np.arange(n_cols) - (n_cols - 1) / 2.0) * dy
    xw = (np.arange(n_rows) - (n_rows - 1) / 2.0) * dx
    XX, YY = np.meshgrid(xw, yw, indexing="ij")
    grid = np.zeros((n_rows, n_cols))
    for (cx, cy, hx, hy, h) in cfg.get("obstacles", []):
        sx = max(hx, 1e-3); sy = max(hy, 1e-3)
        grid += h * np.exp(-(((XX - cx) ** 2) / (2 * sx ** 2)
                             + ((YY - cy) ** 2) / (2 * sy ** 2)))
    return grid, dx, dy, base_y


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
    if TERRAIN_CONFIG.get("kind") not in ("friction", "natural") or _ground_id is None:
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
                       n_rows=256, n_cols=512, height_grid=None,
                       rgba=(0.55, 0.55, 0.6, 1.0), base_xy=(0.0, 0.0)):
    """Build a heightfield body (id 0), calibrated so the start region sits at
    z=0. Either pass a 1-D `height_fn(col_y)` (height varies with Y only,
    constant across X) or a full 2-D `height_grid` of shape (n_rows, n_cols)
    for terrain with lateral structure (rocks, river channels, ...)."""
    if height_grid is not None:
        data = np.asarray(height_grid, dtype=float)
        n_rows, n_cols = data.shape
    else:
        j = np.arange(n_cols)
        col_y = (j - (n_cols - 1) / 2.0) * mesh_dy
        col_h = np.asarray(height_fn(col_y), dtype=float)
        data = np.repeat(col_h[None, :], n_rows, axis=0)      # (n_rows, n_cols)
    # heightfieldData is row-major: index = row + col * n_rows.
    height_list = data.flatten(order="F").astype(np.float64)  # row + col*n_rows

    shape = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        meshScale=[mesh_dx, mesh_dy, 1.0],
        heightfieldData=height_list.tolist(),
        numHeightfieldRows=n_rows,
        numHeightfieldColumns=n_cols,
    )
    terrain = p.createMultiBody(0, shape)
    p.changeVisualShape(terrain, -1, rgbaColor=list(rgba))

    # Place the body (optionally shifted in Y so a long forward landscape needs
    # only a modest grid), then cast a ray at the start (0,0) and shift in Z so
    # the start region sits at z=0.
    bx, by = base_xy
    p.resetBasePositionAndOrientation(terrain, [bx, by, 0], [0, 0, 0, 1])
    surf0 = _surface_z(p, 0.0, 0.0)
    if surf0 is not None:
        p.resetBasePositionAndOrientation(terrain, [bx, by, -surf0], [0, 0, 0, 1])
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
