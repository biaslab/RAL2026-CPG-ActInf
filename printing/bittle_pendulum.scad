/* =====================================================================
   Bittle CoM-Shift  —  PENDULUM / SWEEP ARM  v2
   Base + enclosed servo compartment + corner pivot + threaded-rod arm
   ---------------------------------------------------------------------
   Servo shaft is VERTICAL, so the rod sweeps HORIZONTALLY.
     * gravity contributes NO restoring torque  -> tiny torque demand
     * CoM shift is purely horizontal, no vertical coupling
     * corner pivot means the sweep mixes fore/aft AND lateral (diagonal)

   The threaded rod is the arm. It passes through a tubular boss and is
   clamped by a nut + washer on EACH side, so arm length L is adjustable
   by simply sliding the rod before tightening.

   BOB: use M6 carrosserieringen / penny washers (25mm OD, ~5.4 g each),
   NOT hex nuts. 60 g = 12 penny washers = 18 mm stack.
   60 g of M6 hex nuts would be 24 pieces and a 120 mm stack.

   SWEEP ANGLE: 30 deg is not enough to cross the base. Chord travel is
   2*L*sin(theta/2), so on a 34 mm arm 30 deg buys only ~18 mm of bob
   travel. P1S has 270 deg available -- default here is 90 deg.

   BENCH-VERIFY: shaft_off (shaft position along the servo body) and the
   metal horn's bolt circle. Both are measured, not guessed, next week.
   ===================================================================== */

/* ---------- PART SELECTOR ------------------------------------------ */
part = "layout";   // ["layout","base","lid","arm"]

/* ---------- DECK / ROBOT ------------------------------------------- */
deck_len   = 100;   // MEASURE
deck_wid   = 50;    // MEASURE
robot_mass = 290;   // g, bare Bittle
base_mass  = 40;    // g, est. printed base + lid + arm
bob_mass   = 60;    // g, washer stack  <-- your independent variable

/* ---------- SERVO (P1S: 30 x 24 x 12, 270 deg, 3 kg.cm) ------------ */
servo_l   = 30;
servo_w   = 24;
servo_h   = 12;
shaft_off = 9.5;    // shaft axis from the -X end of the body. MEASURE.
shaft_d   = 5.0;
horn_od   = 16.5;   // metal round horn diameter
horn_th   = 2.0;    // horn thickness (sets arm height above the lid)
horn_bolt_r = 6.0;  // horn bolt circle radius. MEASURE.
horn_bolt_n = 4;

/* ---------- PIVOT PLACEMENT / SWEEP -------------------------------- */
pivot_inset_x = 14; // pivot inset from the -X deck edge
pivot_inset_y = 14; // pivot inset from the -Y deck edge
sweep_start   = 0;  // deg, 0 = rod points +X
sweep_deg     = 90; // 30 is too little; 90..180 is the useful range
arm_len       = 34; // pivot -> bob centre, mm (rod is slidable, so tunable)

/* ---------- ROD / ARM ---------------------------------------------- */
rod_size    = 6;      // M6 threaded rod
rod_bore    = 6.6;    // clearance bore through the boss
boss_r0     = 11;     // boss starts at this radius from the pivot
boss_len    = 14;     // boss bearing length (keeps the rod aligned)
boss_od     = 11;
z_rod       = 8;      // rod axis height above the arm base
hub_d       = 18;
hub_h       = 5;
web_w       = 10;
web_t       = 3;

/* ---------- ENCLOSURE ---------------------------------------------- */
base_th   = 2.5;
wall      = 2.4;
lid_th    = 4.0;
stud_d    = 3.4;      // M3 threaded-rod clearance
nut_af    = 6.2;      // M3 nut across flats
nut_h     = 2.6;      // M3 nut height (recessed FLUSH so the arm clears)
stud_pad  = 9;        // +X flange carrying the two lid studs
cable_w   = 6;
strap_slot_w = 4;
clr       = 0.4;

$fn = 64;

/* ========================= DERIVED ================================= */
px = -deck_len/2 + pivot_inset_x;
py = -deck_wid/2 + pivot_inset_y;

chord   = 2*arm_len*sin(sweep_deg/2);
rod_lin = 0.17;                            // kg/m for M6 threaded rod
rod_m   = rod_lin*(arm_len+25)/1000*1000;  // g, rough
M_total = robot_mass + base_mass + bob_mass + rod_m;
dx_com  = bob_mass*chord/M_total;

cav_x0 = -shaft_off;
cav_x1 =  servo_l - shaft_off;
comp_x0 = cav_x0 - wall;
comp_x1 = cav_x1 + wall + stud_pad;
comp_y0 = -servo_w/2 - wall;
comp_y1 =  servo_w/2 + wall;
comp_z1 = base_th + servo_h;               // top of the compartment walls
lid_z0  = comp_z1;
arm_z0  = lid_z0 + lid_th + horn_th;       // arm base plane
boss_r1 = boss_r0 + boss_len;

stud_x  = cav_x1 + wall + stud_pad/2;
stud_y  = 8;
stud_r  = sqrt(stud_x*stud_x + stud_y*stud_y);

echo(str(">>> sweep ", sweep_deg, " deg, arm L=", arm_len,
         " -> bob chord travel = ", chord, " mm"));
echo(str(">>> CoM shift = ", dx_com, " mm   (M_total ", M_total, " g)"));
echo(str(">>> rod axis sits ", arm_z0 + z_rod, " mm above the deck"));
echo(str(">>> compare: 30 deg would give chord ",
         2*arm_len*sin(15), " mm -> CoM shift ",
         bob_mass*2*arm_len*sin(15)/M_total, " mm"));

// swept-arc containment check against the deck rectangle
module arc_check(){
   for(f=[0,0.5,1]) let(a = sweep_start + f*sweep_deg,
                        tx = px + arm_len*cos(a), ty = py + arm_len*sin(a),
                        on = abs(tx) <= deck_len/2 && abs(ty) <= deck_wid/2)
      echo(str("    bob at ", a, " deg -> (", tx, ", ", ty, ") ",
               on ? "on deck" : "*** OFF DECK ***"));
}
echo(">>> swept arc:");
arc_check();
if (stud_r < boss_r1 + 2)
   echo("!!! lid studs fall inside the arm web radius - move stud_pad out");

/* ========================= BASE ==================================== */
module plate(){
   difference(){
      translate([-deck_len/2, -deck_wid/2, 0])
         cube([deck_len, deck_wid, base_th]);
      for(sx=[-1,1]) translate([sx*(deck_len/2-8), 0, -1])
         cube([strap_slot_w, deck_wid*0.6, base_th+2], center=true);
   }
}

// compartment in LOCAL frame: shaft axis at origin, servo body along +X
module compartment(){
   difference(){
      union(){
         translate([comp_x0, comp_y0, 0])
            cube([comp_x1-comp_x0, comp_y1-comp_y0, comp_z1]);
      }
      // servo cavity, open from the top
      translate([cav_x0-clr, -servo_w/2-clr, base_th])
         cube([servo_l+2*clr, servo_w+2*clr, servo_h+1]);
      // cable exit slot in the +X wall
      translate([cav_x1-1, -cable_w/2, base_th+servo_h-cable_w])
         cube([wall+2, cable_w, cable_w+1]);
      // lid stud holes
      for(sy=[-1,1]) translate([stud_x, sy*stud_y, -1])
         cylinder(d=stud_d, h=comp_z1+2);
   }
}

module base(){
   plate();
   translate([px, py, 0]) compartment();
}

/* ========================= LID ===================================== */
// Nuts are recessed FLUSH into the top face so the arm sweeps clear.
module lid(){
   difference(){
      translate([comp_x0, comp_y0, 0])
         cube([comp_x1-comp_x0, comp_y1-comp_y0, lid_th]);
      translate([0,0,-1]) cylinder(d=horn_od+2, h=lid_th+2);   // shaft/horn clr
      for(sy=[-1,1]){
         translate([stud_x, sy*stud_y, -1]) cylinder(d=stud_d, h=lid_th+2);
         translate([stud_x, sy*stud_y, lid_th-nut_h])
            cylinder(d=nut_af/cos(30), h=nut_h+0.1, $fn=6);    // flush nut pocket
      }
   }
}

/* ========================= ARM ===================================== */
// hub bolts to the metal horn; tubular boss takes the threaded rod,
// clamped by a nut + washer either side -> arm length is adjustable.
module arm(){
   difference(){
      union(){
         cylinder(d=hub_d, h=hub_h);                            // hub
         translate([0, -web_w/2, 0]) cube([boss_r1, web_w, web_t]); // web plate
         translate([boss_r0, -web_w/2, web_t-0.01])              // boss support
            cube([boss_len, web_w, z_rod-web_t+0.01]);
         translate([boss_r0, 0, z_rod]) rotate([0,90,0])         // tubular boss
            cylinder(d=boss_od, h=boss_len);
      }
      translate([0,0,-1]) cylinder(d=shaft_d+0.6, h=hub_h+2);    // shaft bore
      translate([0,0,-0.01]) cylinder(d=horn_od+0.5, h=horn_th+0.3); // horn seat
      for(k=[0:horn_bolt_n-1]) rotate(k*360/horn_bolt_n)
         translate([horn_bolt_r,0,-1]) cylinder(d=2.2, h=hub_h+2);
      translate([boss_r0-1, 0, z_rod]) rotate([0,90,0])          // rod bore
         cylinder(d=rod_bore, h=boss_len+2);
   }
}

/* ========================= LAYOUT ================================== */
module bob_stack(){
   // 60 g of M6 penny washers ~= 18 mm of 25 mm OD stack
   color("dimgray") rotate([0,90,0]) cylinder(d=25, h=18, center=true);
}

module layout(){
   color("lightsteelblue") plate();
   translate([px, py, 0]){
      color("gainsboro") compartment();
      color("silver")   translate([0,0,lid_z0]) lid();
      rotate([0,0,sweep_start]) translate([0,0,arm_z0]){
         color("khaki") arm();
         color("goldenrod") translate([boss_r0+2, 0, z_rod])   // the rod
            rotate([0,90,0]) cylinder(d=rod_size, h=arm_len+20);
         translate([arm_len, 0, z_rod]) bob_stack();
      }
      // ghost of the far end of the sweep
      rotate([0,0,sweep_start+sweep_deg]) translate([0,0,arm_z0])
         color([0.9,0.7,0.2,0.25]){
            translate([boss_r0+2, 0, z_rod]) rotate([0,90,0])
               cylinder(d=rod_size, h=arm_len+20);
            translate([arm_len, 0, z_rod]) rotate([0,90,0])
               cylinder(d=25, h=18, center=true);
         }
   }
}

if      (part=="layout") layout();
else if (part=="base")   base();
else if (part=="lid")    lid();
else if (part=="arm")    arm();

/* =====================================================================
   BOM
     1x P1S servo + metal round horn, 4x M2 screws (arm -> horn)
     M6 threaded rod, ~arm_len + 30 mm
     4x M6 nuts (2 clamp the boss, 2 clamp the bob) + 2x M6 plain washers
     ~12x M6 carrosseriering / penny washer, 25 mm OD  (= 60 g bob)
     2x M3 studs ~ (comp_z1 + lid_th) long + 2x M3 nuts  (lid clamp)
     velcro or zip ties through the strap slots

   PRINTING (Bambu, PLA)
     base : plate-down, 0.20mm, 3-4 walls, 15-20% gyroid. The compartment
            cavity is vertical-walled -> no supports.
     lid  : flat, 0.16mm, 4 walls, ~40% infill. Hex nut pockets print as
            clean vertical prisms.
     arm  : web-down (as modelled), 0.16mm, 4 walls, 40-60% infill. The
            boss bore prints horizontally -> ream with a 6.5mm bit or
            just run the rod through to clear it.

   ASSEMBLY
     1. Servo into the compartment, cable out the slot, lid on, 2x M3
        studs + nuts recessed flush.
     2. Metal horn on the shaft. Bolt the arm to the horn (4x M2).
     3. Rod through the boss. Nut + washer either side, tightened. Slide
        the rod first to set arm_len -- this is your CoM authority knob.
     4. Washer stack on the far end between two nuts.
     5. CENTRE THE SERVO IN SOFTWARE FIRST, then fit the arm at the
        middle of the intended sweep so you have +-sweep_deg/2 available.

   TWO THINGS TO CHECK ON THE ROBOT
     - LEG CLEARANCE. The rod axis sits ~28 mm above the deck; the bob is
       25 mm across. Sweeping over the body corners it may foul the
       shoulder servos. Set software end stops before commanding a sweep.
     - The arc leaves the deck footprint if arm_len is too large -- the
       echo above checks this once you enter real deck dimensions.

   HEIGHT NOTE: this stacks base + servo + lid + horn, putting the rod
   ~28 mm up, which raises the CoM on an already tippy robot. If the deck
   has room underneath, cut a hole in the plate and drop the servo
   through it to recover ~12 mm.
   ===================================================================== */
