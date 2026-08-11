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

part = "layout";   // ["layout","base","arm"]

/* ---------- MOUNTING (measured) ------------------------------------ */
hole_dx    = 60;
hole_dy    = 50;
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
horn_bolt_n = 4;
horn_screw_d   = 1.7;  // self-tapping M2 into PLA (blind, from above)
horn_screw_len = 6.0;
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
   The rod is NOT clamped between two nuts - the bore is blind, so there
   is no inner face. Instead an M6 nut is trapped in a hex pocket at the
   OUTER end of the bore and the rod threads into it. Real threads, and
   arm_len stays adjustable by screwing the rod in or out.               */
z_rod    = 15;      // rod axis above the plate. Floor set by bob radius.
rod_bore = 6.6;     // free-running bore inboard of the nut
slab_r0  = 5;
slab_r1  = 25;      // slab / bore outer end
slab_w   = 10;
slab_h   = 11;
hub_d    = 18;
hub_up   = 8;
// M6 nut trap at the outer end, in a local boss (a nut is too big to
// fit inside a 10 x 11 slab, hence the boss)
nut6_af  = 10.2;    // M6 nut across flats + clearance
nut6_th  = 5.5;     // M6 nut thickness + clearance
boss_x   = 15;      // boss cross-section, square, centred on the bore

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
plat_th  = 5;
win_pad  = 28;      // window opens this many deg beyond the sweep
rib_w    = 3;
rib_h    = 8;
rib_r0   = hub_d/2 + 2;   // must clear the rotating arm hub
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

// smallest arm length that keeps the stack clear of the slab and its nut
arm_len_min = slab_r1 + nut6_th + bob_len/2 + 1;
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
   the window. Two ribs in the un-swept sector break up the platform's
   underside bridge. NOTE the near tab screw lands over the bridged
   region, not over a wall - 5 mm of PLA under a 12 g servo is fine,
   but do not overtighten it.                                            */
module pedestal(){
   union(){
      // shell
      difference(){
         cylinder(r=r_out, h=h_shell);
         translate([0,0,-1]) cylinder(r=r_in, h=h_shell+2);
         translate([0,0,-0.5]) sector(win_a0, win_a1, r_out+6, h_shell+1);
      }
      // ribs, in the un-swept sector, under the platform. They start
      // OUTSIDE the arm hub radius or they would clash with it.
      for(a=[win_a1+45, win_a1+90])
         rotate([0,0,a]) translate([rib_r0, -rib_w/2, h_shell-rib_h])
            cube([r_in-rib_r0+0.1, rib_w, rib_h]);
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
         translate([0, -slab_w/2, z0]) cube([slab_r1, slab_w, slab_h]);
         // local boss at the outer end to host the trapped M6 nut
         translate([slab_r1-boss_x*0.4, -boss_x/2, z_rod-boss_x/2])
            cube([boss_x*0.4, boss_x, boss_x]);
      }
      // central bore: driver access from below for the horn's shaft screw
      translate([0,0,z0-1]) cylinder(d=shaft_d+0.6, h=arm_top-z0+2);
      // horn seat
      translate([0,0,pkz]) cylinder(d=horn_od+0.5, h=horn_th+1);
      // BLIND self-tapping holes. Horn screws go DOWN from above, on the
      // bench, before the arm ever goes near the robot.
      for(k=[0:horn_bolt_n-1]) rotate([0,0,k*360/horn_bolt_n])
         translate([horn_bolt_r, 0, pkz-horn_screw_len])
            cylinder(d=horn_screw_d, h=horn_screw_len+1);
      // rod bore, horizontal, free-running inboard of the nut
      translate([slab_r0, 0, z_rod]) rotate([0,90,0])
         cylinder(d=rod_bore, h=slab_r1-slab_r0+2);
      // HEX NUT TRAP, open at the outer face: slide an M6 nut in from
      // outside, then thread the rod into it. The shoulder at the inner
      // end of the pocket takes the load.
      translate([slab_r1-nut6_th, 0, z_rod]) rotate([0,90,0])
         cylinder(d=nut6_af/cos(30)+0.2, h=nut6_th+2, $fn=6);
   }
}

/* ========================= LAYOUT ================================== */
module arm_and_rod(){
   color("khaki") arm();
   color("goldenrod") translate([slab_r0+2, 0, z_rod]) rotate([0,90,0])
      cylinder(d=6, h=L+bob_len/2);
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
            top of it, which is the face that matters. The two ribs
            shorten the worst of the span.
     arm  : as modelled, slab-down. 0.16mm, 5 walls, 60% infill. This is
            the whole load path. The rod bore prints horizontally, so
            ream it with a 6.5mm bit or drive the rod through.

   ASSEMBLY  (order matters - see note below)
     1. BENCH, arm off the robot: drop the metal horn into the arm's top
        pocket and drive the 4 horn screws DOWNWARD into the blind holes.
        Full access, nothing in the way. Arm + horn is now one piece.
     2. Fit the rod through the slab now too, nut + washer each side.
        Slide it to set arm_len before tightening. Easier on the bench
        than in situ. Leave the washer stack off for now - less mass to
        hold while you align the spline.
     3. Centre the servo in software.
     4. Servo down through the platform hole from above, tabs onto the
        platform top, 2 tab screws through the slots.
     5. Bring the arm+horn subassembly in through the pedestal WINDOW and
        lift it onto the shaft, engaging the spline at mid-sweep. There is
        ~10 mm of vertical play to work with.
     6. Drive the horn's central retaining screw UP into the shaft, with a
        driver through the access hole in the plate. This is the ONLY
        from-below operation, and the only reason that hole exists.
     7. Washer stack on the rod end between two nuts (nylon locknut
        outside).
     8. Plate onto the board on the metal spacers.
     9. Sweep by hand through the full range checking LEG CLEARANCE
        before powering the servo.

   WHY THE ACCESS HOLE EXISTS
     The plate and pedestal are one printed part, so the plate sits 9.5 mm
     under the arm permanently - there is no assembly order that gets a
     driver in from below. Two changes fix it: the horn screws are now
     BLIND and driven downward on the bench (step 1), and the single
     central shaft screw is reached through the access hole (step 6).
     If your horn has NO central retaining screw, it relies on spline
     friction - fit it firmly and check it after the first few sweeps.

   WHY THE ROD SITS AT 15mm
     The bob is 18mm in DIAMETER, so it hangs 9mm below the rod axis.
     z_rod must exceed bob_od/2 + base_th + clearance. If you use a
     smaller-diameter bob you can lower z_rod and win back CoM height -
     the echo checks this for you.
   ===================================================================== */
