/* =====================================================================
   Bittle CoM-Shift  —  PENDULUM, INVERTED SERVO  v4
   ---------------------------------------------------------------------
   Corrected servo orientation: the P1S 30 mm dimension is ALONG THE
   SHAFT. So it is a 24 x 12 footprint, 30 mm tall - SG90 form factor.

   The servo is mounted UPSIDE DOWN: shaft pointing DOWN through a
   pedestal platform, body standing as a tower above it, arm sweeping
   low just above the mounting plate. This keeps the heavy 60 g bob at
   ~15 mm above the plate instead of ~48 mm.

   Servo is held by its OWN MOUNTING TABS bolted to the platform top -
   the standard way, stronger than any clamp. So there is no lid and no
   compartment: only TWO printed parts, base and arm.

   ONLY 2 PRINTED PARTS:  base (plate + pedestal + platform),  arm

   ---------------------------------------------------------------------
   MEASURE THESE FOUR. Everything else is derived.
     shaft_off_a    where the shaft sits along the 24 mm dimension
     shaft_stickout tab mounting plane -> tip of the shaft
     tab_screw_d    servo tab screw diameter
     case_below_tab case thickness between tab plane and shaft face
                    (must be < plat_th or the case fouls the platform)
   The tab screw holes are SLOTS, so exact tab hole spacing does not
   need to be known - that is deliberate.
   ===================================================================== */

part = "arm";   // ["layout","base","arm"]

/* ---------- MOUNTING (measured) ------------------------------------ */
hole_dx    = 58;
hole_dy    = 49;
screw_d    = 2.4;   // MEASURE
cb_d       = 5.5;
cb_h       = 2.0;
standoff_h = 8.0;   // your metal spacers
plate_edge = 5.0;
base_th    = 3.5;

/* ---------- SERVO: 24 x 12 footprint, 30 tall along the shaft ------- */
servo_a   = 24;     // footprint, long
servo_b   = 12;     // footprint, short
servo_len = 30;     // ALONG THE SHAFT
shaft_off_a    = 6; // MEASURE: shaft from the -A end of the 24mm dim
shaft_stickout = 9; // MEASURE: tab plane -> shaft tip
case_below_tab = 4; // MEASURE: must stay below plat_th
tab_screw_d    = 2.4;
serv_ang  = 225;    // servo long axis, pointed into the un-swept sector

/* ---------- HORN --------------------------------------------------- */
horn_od     = 16.5;
horn_th     = 2.0;
horn_bolt_r = 6.0;  // MEASURE
horn_bolt_n = 2;    // 2 only, and at +-90 deg: see the ARM note below
horn_screw_d   = 1.7;  // self-tapping M2 into PLA (blind, from above)
horn_screw_len = 6.0;
shaft_screw_head_d = 7.0;  // central shaft screw head + washer clearance
shaft_screw_head_h = 3.0;  // MEASURE if your screw has a big head
shaft_d     = 5.0;

/* ---------- ASSEMBLY ACCESS ----------------------------------------
   Hole through the plate/shield at the pivot so a driver can reach the
   horn's central retaining screw from underneath. Without this the arm
   cannot be fitted at all - the plate blocks the only approach.
   Sits at r < 10, so it does not weaken the shell footprint (r_in=19),
   nor open the shield where the bob travels (r 18..50).                */
access_d = 18;

/* ---------- PIVOT / SWEEP ------------------------------------------ */
pivot_x     = -12;
pivot_y     = 0;
sweep_start = 0;    // 0 = diagonal, -45 = lateral, +45 = fore/aft
sweep_deg   = 90;
arm_len     = 0;    // 0 = AUTO: smallest L that clears the slab + nut

/* ---------- ARM / ROD ----------------------------------------------
   The rod passes RADIALLY THROUGH the arm, straight across the pivot,
   and is clamped by a nut + washer on EACH face. Positively retained:
   even if a thread lets go the rod cannot leave the arm.

   Two consequences of routing it through the middle:
     - the rod crosses the vertical driver bore, so it must be fitted
       LAST, after the horn's central shaft screw is done up
     - the horn takes 2 screws at +-90 deg, PERPENDICULAR to the rod.
       A 4-screw pattern would put one straight through the bore.       */
z_rod    = 15;      // rod axis above the plate. Floor set by bob radius.
rod_bore = 6.6;
slab_in  = 10;      // slab reaches this far to -X, giving the inner face
slab_r1  = 25;      // slab outer end
slab_w   = 10;
slab_h   = 11;
hub_d    = 18;
hub_up   = 15;      // tall on purpose: lifts the horn and its central
                    // screw clear of the rod passing under it. 13.3 is the
                    // hard minimum - below that the screw head fouls the rod.
nut6_th  = 5.5;     // M6 nut thickness, for the arm_len clearance sum
wash6_th = 1.6;     // M6 plain washer (OUTER side only)
nut6_af_c = 11.8;   // M6 nut across corners - sets the swing radius

/* ---------- BOB: derived from the washers you actually buy -----------
   M6 carrosseriering (DIN 9021): 18 mm OD, 1.6 mm, 2.79 g
   M8 carrosseriering            : 25 mm OD, 2.0 mm, 6.84 g  <- shorter
     stack for the same mass, so a shorter arm and a much smaller
     shield, but needs z_rod >= 17. Loose on M6 rod; clamp between two
     plain M6 washers.                                                  */
washer_od = 18;
washer_th = 1.6;
washer_g  = 2.79;
washer_n  = 20;

/* ---------- PEDESTAL ----------------------------------------------- */
r_in     = 19;
r_out    = 23;
r_plat   = 26;
plat_th  = 6;      // 6 not 5: no ribs now, so the bridge needs the depth
win_pad  = 28;      // window opens this many deg beyond the sweep
wall     = 2.4;
clr      = 0.4;

/* ---------- PROTECTIVE SHIELD --------------------------------------
   Flat shelf at plate level, under the bob's swept path, so nothing
   can reach the battery below. Defaults cover the whole sweep; trim
   shield_a0/a1/r if you only need to guard one region.                 */
shield_on = false;
shield_th = 2.5;
shield_r  = 0;      // 0 = auto (bob outer radius + margin)
shield_a0 = 0;      // 0/0 = auto (matches the swept sector)
shield_a1 = 0;

/* ---------- MASSES (for the echo only) ----------------------------- */
robot_mass = 290;
print_mass = 50;
// bob_mass is derived from the washer spec (see below)

$fn = 72;

/* ========================= DERIVED ================================= */
plate_len = hole_dx + 2*plate_edge;
plate_wid = hole_dy + 2*plate_edge;

arm_top    = z_rod + hub_up;          // = shaft tip height
h_plat_top = arm_top + shaft_stickout;
h_shell    = h_plat_top - plat_th;

// bob geometry and mass follow from the washer spec
bob_od   = washer_od;
bob_len  = washer_n * washer_th;
bob_mass = washer_n * washer_g;

// smallest arm length that keeps the stack clear of the slab, plus the
// outer clamping nut and washer
arm_len_min = slab_r1 + nut6_th + wash6_th + bob_len/2 + 1;
L = arm_len > 0 ? arm_len : arm_len_min;

chord   = 2*L*sin(sweep_deg/2);
rod_m   = 0.17*(L + bob_len/2 + 20);
M_total = robot_mass + print_mass + bob_mass + rod_m;
dx_com  = bob_mass*chord/M_total;

win_a0 = sweep_start - win_pad;
win_a1 = sweep_start + sweep_deg + win_pad;

// shield: auto-size to the bob's swept path unless overridden
sh_r  = shield_r  > 0 ? shield_r  : L + bob_len/2 + 4;
sh_a0 = (shield_a0 == 0 && shield_a1 == 0) ? win_a0 : shield_a0;
sh_a1 = (shield_a0 == 0 && shield_a1 == 0) ? win_a1 : shield_a1;

// how far past each plate edge does the shield reach?
function sh_x(a) = pivot_x + sh_r*cos(a);
function sh_y(a) = pivot_y + sh_r*sin(a);
sh_samp = [for(i=[0:36]) sh_a0 + (sh_a1-sh_a0)*i/36];
sh_xmax = max([for(a=sh_samp) sh_x(a)]);
sh_xmin = min([for(a=sh_samp) sh_x(a)]);
sh_ymax = max([for(a=sh_samp) sh_y(a)]);
sh_ymin = min([for(a=sh_samp) sh_y(a)]);

echo(str(">>> arm top / shaft tip : ", arm_top, " mm above plate"));
echo(str(">>> platform top (tabs) : ", h_plat_top, " mm"));
echo(str(">>> servo body top      : ", h_plat_top + servo_len - shaft_stickout, " mm"));
echo(str(">>> bob spans z ", z_rod - bob_od/2, " .. ", z_rod + bob_od/2,
         " ; plate top ", base_th, " -> clearance ", z_rod - bob_od/2 - base_th, " mm"));
if (z_rod - bob_od/2 - base_th < 1)
   echo("!!! BOB WILL HIT THE PLATE - raise z_rod or use a smaller bob_od");
if (case_below_tab > plat_th)
   echo("!!! case_below_tab > plat_th: servo case will foul the platform hole");
// the rod's inner nut sweeps the sector opposite the arm; it must stay
// inside the pedestal shell's bore
inner_nut_r = sqrt(pow(slab_in+nut6_th,2) + pow(nut6_af_c/2,2));
// vertical clearance between the rod passing through and the horn
// hardware sitting above it
pkz_z     = arm_top - horn_th - 0.5;
bore_top  = z_rod + rod_bore/2;
head_gap  = pkz_z - shaft_screw_head_h - bore_top;
hscrew_gap= pkz_z - horn_screw_len - bore_top;
echo(str(">>> rod bore top z=", bore_top,
         " ; screw head pocket starts z=", pkz_z - shaft_screw_head_h,
         " -> gap ", head_gap, " mm"));
echo(str(">>> horn screws bottom out at z=", pkz_z - horn_screw_len,
         " -> gap ", hscrew_gap, " mm"));
if (head_gap < 1.5 || hscrew_gap < 1.5)
   echo("!!! HORN HARDWARE FOULS THE ROD - raise hub_up (needs > 13.3)");

echo(str(">>> rod passes right through: ", slab_in+slab_r1,
         " mm of bore, nut on each face"));
echo(str(">>> inner nut swings to r=", inner_nut_r,
         " ; shell bore r_in=", r_in));
if (inner_nut_r > r_in - 1.5)
   echo("!!! INNER NUT FOULS THE SHELL - raise r_in/r_out/r_plat, or drop slab_in");
echo("    NOTE: inner side takes a BARE NUT, no washer. A washer or nylon");
echo("    locknut there adds ~1.6mm of radius and will not clear r_in.");
echo(str(">>> bob: ", washer_n, " x ", washer_od, "mm washers = ", bob_mass,
         " g in a ", bob_len, " mm stack"));
echo(str(">>> arm_len: auto minimum is ", arm_len_min,
         " mm (slab ends ", slab_r1, " + nut ", nut6_th, " + half stack)"));
if (arm_len > 0 && arm_len < arm_len_min)
   echo("!!! arm_len TOO SHORT - the washer stack would sit inside the arm");
if (z_rod - bob_od/2 - shield_th < 1)
   echo("!!! BOB WILL FOUL THE SHIELD - raise z_rod or use smaller washers");
echo(str(">>> sweep ", sweep_deg, " deg, L=", L, (arm_len>0?" (manual)":" (auto)"), " -> chord ", chord, " mm"));
echo(str(">>> CoM shift = ", dx_com, " mm  (M_total ", M_total, " g)"));
echo(str(">>> window ", win_a0, " .. ", win_a1, " deg; servo axis at ", serv_ang));
if (shield_on){
   echo(str(">>> shield: r=", sh_r, " mm, ", sh_a0, " .. ", sh_a1, " deg"));
   echo(str("    reaches x ", sh_xmin, " .. ", sh_xmax,
            " ; plate is ", -plate_len/2, " .. ", plate_len/2));
   echo(str("    reaches y ", sh_ymin, " .. ", sh_ymax,
            " ; plate is ", -plate_wid/2, " .. ", plate_wid/2));
   echo(str("    OVERHANG past plate edge:  +x ",
            max(0, sh_xmax - plate_len/2), "  -x ", max(0, -plate_len/2 - sh_xmin),
            "  +y ", max(0, sh_ymax - plate_wid/2),
            "  -y ", max(0, -plate_wid/2 - sh_ymin), " mm"));
   echo("    ^ CHECK THESE AGAINST THE LEGS. To shrink: lower arm_len, or");
   echo("      set shield_a0/a1/shield_r to guard only where the battery is.");
}

/* ========================= HELPERS ================================= */
// pie-slice prism, used to cut the window out of the shell
module sector(a0, a1, r, h){
   linear_extrude(h)
      polygon(concat([[0,0]],
         [for(i=[0:24]) let(a = a0 + (a1-a0)*i/24) [r*cos(a), r*sin(a)]]));
}

/* ========================= MOUNTING PLATE ========================== */
module mount_plate(){
   difference(){
      translate([-plate_len/2, -plate_wid/2, 0])
         cube([plate_len, plate_wid, base_th]);
      for(sx=[-1,1]) for(sy=[-1,1])
         translate([sx*hole_dx/2, sy*hole_dy/2, 0]){
            translate([0,0,-1]) cylinder(d=screw_d, h=base_th+2);
            translate([0,0,base_th-cb_h]) cylinder(d=cb_d, h=cb_h+0.1);
         }
   }
}

/* ========================= PEDESTAL ================================
   Cylindrical shell with a window cut in the swept sector, topped by a
   full disc platform. The arm sweeps inside the shell and out through
   the window. There are deliberately NO internal ribs - the rod's inner
   nut sweeps the sector opposite the arm, so anything projecting inward
   would be struck. The platform is 6 mm thick to carry the bridge span
   on its own. The near tab screw lands over that bridged region rather
   than over a wall; fine under a 12 g servo, but do not overtighten.    */
module pedestal(){
   // shell only. NO internal ribs: the rod's INNER nut sweeps the sector
   // opposite the arm (roughly 155..295 deg) at radius ~16, so anything
   // projecting inward from the wall gets hit.
   difference(){
      cylinder(r=r_out, h=h_shell);
      translate([0,0,-1]) cylinder(r=r_in, h=h_shell+2);
      translate([0,0,-0.5]) sector(win_a0, win_a1, r_out+6, h_shell+1);
   }
}

module platform(){
   difference(){
      cylinder(r=r_plat, h=plat_th);
      // servo case passes down through here; shaft continues below
      rotate([0,0,serv_ang])
         translate([-shaft_off_a-clr, -(servo_b+2*clr)/2, -1])
            cube([servo_a+2*clr, servo_b+2*clr, plat_th+2]);
      // tab screw SLOTS either side, so tab spacing need not be known
      rotate([0,0,serv_ang]) for(s=[-1,1])
         hull() for(d=[1,8])
            translate([s>0 ? (servo_a-shaft_off_a)+d : -shaft_off_a-d, 0, -1])
               cylinder(d=tab_screw_d, h=plat_th+2);
   }
}

module shield(){
   sector(sh_a0, sh_a1, sh_r, shield_th);
}

// plate + shield, with the driver access hole cut through both
module plate_assembly(){
   difference(){
      union(){
         mount_plate();
         if(shield_on) translate([pivot_x, pivot_y, 0]) shield();
      }
      translate([pivot_x, pivot_y, -1])
         cylinder(d=access_d, h=max(base_th, shield_th)+2);
   }
}

module base(){
   plate_assembly();
   translate([pivot_x, pivot_y, base_th]){
      pedestal();
      translate([0,0,h_shell]) platform();
   }
}

/* ========================= ARM =====================================
   Hangs from the horn below the platform. Hub column on top, slab with
   a through bore for the M6 rod at the bottom. Rod is clamped by a nut
   and washer either side, so arm_len stays adjustable.                 */
module arm(){
   z0  = z_rod - slab_h/2;          // slab bottom = lowest point of the arm
   pkz = arm_top - horn_th - 0.5;   // horn pocket floor
   difference(){
      union(){
         translate([0,0,z0]) cylinder(d=hub_d, h=arm_top-z0);
         // slab spans BOTH sides of the pivot so the rod has a face at
         // each end for a clamping nut
         translate([-slab_in, -slab_w/2, z0])
            cube([slab_in+slab_r1, slab_w, slab_h]);
      }
      // vertical driver bore: reaches the horn's central shaft screw.
      // The rod crosses it, so the rod goes in LAST.
      translate([0,0,z0-1]) cylinder(d=shaft_d+0.6, h=arm_top-z0+2);
      // horn seat
      translate([0,0,pkz]) cylinder(d=horn_od+0.5, h=horn_th+1);
      // COUNTERBORE for the central shaft screw's head + washer. It sits
      // UNDER the horn, pressing up on it, so it needs its own pocket.
      // This is what was fouling the rod before the hub was raised.
      translate([0,0,pkz-shaft_screw_head_h])
         cylinder(d=shaft_screw_head_d, h=shaft_screw_head_h+0.1);
      // horn screws: blind, driven DOWN on the bench, and offset 90 deg
      // so they sit either side of the rod bore rather than in it
      for(k=[0:horn_bolt_n-1]) rotate([0,0,90 + k*360/horn_bolt_n])
         translate([horn_bolt_r, 0, pkz-horn_screw_len])
            cylinder(d=horn_screw_d, h=horn_screw_len+1);
      // rod bore, RADIAL and fully through
      translate([-slab_in-2, 0, z_rod]) rotate([0,90,0])
         cylinder(d=rod_bore, h=slab_in+slab_r1+4);
   }
}

/* ========================= LAYOUT ================================== */
module arm_and_rod(){
   color("khaki") arm();
   color("goldenrod") translate([-slab_in-6, 0, z_rod]) rotate([0,90,0])
      cylinder(d=6, h=slab_in+L+bob_len/2+8);
   color("dimgray") translate([L, 0, z_rod]) rotate([0,90,0])
      cylinder(d=bob_od, h=bob_len, center=true);
}

module servo_ghost(){
   rotate([0,0,serv_ang]) translate([-shaft_off_a, -servo_b/2, 0])
      cube([servo_a, servo_b, servo_len - shaft_stickout + case_below_tab]);
}

module layout(){
   color("lightsteelblue") plate_assembly();
   translate([pivot_x, pivot_y, base_th]){
      color("gainsboro") pedestal();
      color("silver") translate([0,0,h_shell]) platform();
      color([0.4,0.5,0.6,0.35]) translate([0,0,h_plat_top-case_below_tab])
         servo_ghost();
      rotate([0,0,sweep_start]) arm_and_rod();
      rotate([0,0,sweep_start+sweep_deg]) color([0.9,0.7,0.2,0.25])
         arm_and_rod();
   }
}

if      (part=="layout") layout();
else if (part=="base")   base();
else if (part=="arm")    arm();

/* =====================================================================
   BOM
     1x P1S servo + its 2 tab screws + metal round horn + 4x M2
     M6 threaded rod, ~arm_len + 30 mm
     4x M6 nuts (2 clamp the slab, 2 clamp the bob) + 2 plain washers
     ~20x M6 carrosseriering 18 mm OD  (= 56 g bob)
     4x metal spacers + screws for the plate

   PRINTING (Bambu, PLA)
     base : plate-down. 0.20mm, 4 walls, 20% gyroid. No supports.
            The platform's UNDERSIDE bridges the shell interior (~38mm).
            Enable bridge settings; the underside will look rough but
            the TOP face - the servo mounting surface - prints flat on
            top of it, which is the face that matters.
     arm  : as modelled, slab-down. 0.16mm, 5 walls, 60% infill. This is
            the whole load path. The rod bore prints horizontally, so
            ream it with a 6.5mm bit or drive the rod through.

   ASSEMBLY  (order matters - the rod goes in LAST)
     1. BENCH: drop the metal horn into the arm's top pocket and drive the
        2 horn screws DOWNWARD into the blind holes. Use the two holes
        that sit either side of the rod bore. Arm + horn is now one piece.
     2. Centre the servo in software.
     3. Servo down through the platform hole from above, tabs onto the
        platform top, 2 tab screws through the slots.
     4. Bring the arm+horn subassembly in through the pedestal WINDOW and
        lift it onto the shaft, engaging the spline at mid-sweep.
     5. Drive the horn's central retaining screw UP into the shaft, with a
        driver through the access hole in the plate and up the arm's
        vertical bore. DO THIS BEFORE THE ROD GOES IN - the rod crosses
        that bore and will block the driver.
     6. Slide the rod through the arm. Nut + washer on the INNER face and
        on the OUTER face, tightened against each other. The rod is now
        positively trapped: it cannot back out even if a thread fails.
        Slide it before tightening to set the arm length.
     7. Washer stack on the outer end between two nuts, nylon locknut
        outermost.
     8. Plate onto the board on the metal spacers.
     9. Sweep by hand through the full range checking LEG CLEARANCE
        before powering the servo.

   NOTE: re-tightening the horn's central screw later means taking the rod
   out first. Do it up properly at step 5.

   WHY THE ACCESS HOLE EXISTS
     The plate and pedestal are one printed part, so the plate sits under
     the arm permanently - there is no assembly order that gets a driver
     in from below. The horn screws are BLIND and driven downward on the
     bench (step 1); only the central shaft screw needs the hole (step 5).
     If your horn has NO central retaining screw, it relies on spline
     friction - fit it firmly and check it after the first few sweeps.

   WHY THE ROD SITS AT 15mm
     The bob is 18mm in DIAMETER, so it hangs 9mm below the rod axis.
     z_rod must exceed bob_od/2 + base_th + clearance. If you use a
     smaller-diameter bob you can lower z_rod and win back CoM height -
     the echo checks this for you.
   ===================================================================== */
