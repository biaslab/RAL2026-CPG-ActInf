---
name: deformable-terrain-options
description: pavement->grass deformable-terrain idea; PyBullet soft-body support facts; compliant-contact (changeDynamics contactStiffness) is INERT for our light robot — measured <1cm sink
metadata: 
  node_type: memory
  type: project
  originSessionId: 0f8f0747-1c9d-4007-96d6-8da8b667cf75
---

Idea (2026-07-13): single pavement->grass transition; soft ground = a genuinely different STEADY-STATE
dynamics regime (not transition shock), so it's the most promising scenario to get oracle > no-adapt.

**PyBullet deformable support (installed build 202010061):** HAS loadSoftBody, createSoftBodyAnchor,
RESET_USE_DEFORMABLE_WORLD, RESET_USE_REDUCED_DEFORMABLE_WORLD. BUT soft bodies are mesh(.obj)-based,
10-100x slower, fragile (rigid-foot-into-softbody penetration bugs, collision-margin/self-penetration
issues). Literature consensus: PyBullet does NOT model soft/granular terrain at locomotion scale; groups
use Project Chrono or custom granular/surrogate models (e.g. Science Robotics "Learning quadrupedal
locomotion on deformable terrain"). Full soft-body would also require rewriting our rigid-heightfield
/friction-zone/fall-check pipeline. High risk of a quagmire.

**Compliant-contact surrogate (changeDynamics contactStiffness/contactDamping) — TRIED, FAILED (inert):**
Incumbent gait on flat ground is UNAFFECTED across stiffness 1e5 -> 5e2 (0/3 falls, vx~0.6, tip<1° at all).
Diagnosis: foot sink is negligible — min foot z only 0.023 (rigid) -> 0.013 (ultra-soft 1e2), base z
0.402 -> 0.394 (<1cm). The 10kg robot on small point feet under POSITION control doesn't press hard enough
to sink; penalty contact only affects normal penetration, not the energy-absorption/shear that makes real
soft ground hard. So contactStiffness cannot create a soft-ground effect for this platform.

**Remaining options if pursuing soft ground:** (a) full loadSoftBody (real deformation, high cost/risk,
may not scale to 100 seeds); (b) a surrogate SOIL FORCE model — custom resistive+damping force on stance
feet (penetration spring + velocity damper + load-dependent traction loss), cheap & in-pipeline but custom
physics needing citation (granular RFT / surrogate-compliance papers); (c) reconsider. Not yet chosen.
NOTE: given incumbent+VMC feedback robustness across ALL prior terrains [[oracle-cannot-beat-noadapt]]
[[gain-adaptation-channel]], skepticism warranted that any surrogate cleanly yields oracle>no-adapt.
