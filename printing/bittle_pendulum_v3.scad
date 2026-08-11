/* =====================================================================
   Bittle CoM-Shift  —  PENDULUM / SWEEP ARM  v3
   Now BOLTS DOWN to the electronics stack via a 60 x 50 screw pattern
   ---------------------------------------------------------------------
   v3 changes:
     - strap/velcro slots REMOVED. The plate now bolts to the 4 mounting
       screws of the electronics stack (measured 60 x 50 mm).
     - plate stands on 4 FEET so it clears components on the PCB.
     - screw heads are counterbored FLUSH so the arm sweeps over them.
     - pivot moved onto the plate centreline (y = 0), which both clears
       the mounting bosses and makes sweep_start a clean axis selector.

   MEASURE THESE TWO before printing:
     screw_d     -- Petoi mostly uses M2, but check yours
     standoff_h  -- must clear the TALLEST component/connector on the
                    board, including any plugged-in cable. Default 10 mm
                    is a guess. Every mm here also raises the CoM, so
                    take the smallest value that actually clears.

   SWEEP AXIS: with the pivot on the centreline, sweep_start picks which
   axis the payload shifts along (all with sweep_deg = 90, L = 34):
        sweep_start =   0  -> diagonal      (dx -34, dy +34)
        sweep_start = -45  -> pure lateral  (dy +48)
        sweep_start =  45  -> pure fore/aft (dx -48)
   Default is 0 = the diagonal you originally asked for.

   Bolt loads are a non-issue: at servo stall the reaction couple across
   the 78 mm bolt diagonal is 3.8 N total, under 1 N per screw.
   ===================================================================== */

/* ---------- PART SELECTOR ------------------------------------------ */
part = "layout";   // ["layout","base","lid","arm","spacer"]

/* Integral feet print BADLY: the plate underside becomes a standoff_h
   high unsupported overhang and the feet are the only bed contact.
   Leave this false and print 4 separate spacers (part="spacer"), which
   print perfectly standing on end. Or use nylon standoffs.             */
integral_feet = false;
underside_relief = 0;   // 0 = flat underside. Using metal spacers, leave at 0.

/* ---------- MOUNTING (measured) ------------------------------------ */
hole_dx    = 60;    // MEASURED screw spacing, long direction
hole_dy    = 50;    // MEASURED screw spacing, short direction
screw_d    = 2.4;   // M2 clearance. MEASURE - could be M2.5 or M3.
cb_d       = 5.5;   // counterbore for the screw head
cb_h       = 2.0;   // counterbore depth (head sits flush)
foot_od    = 8.0;   // standoff foot diameter
standoff_h = 10.0;  // MEASURE: clearance over the tallest component
plate_edge = 5.0;   // plate margin outside the screw pattern

/* ---------- ROBOT / MASS ------------------------------------------- */
robot_mass = 290;   // g, bare Bittle
base_mass  = 45;    // g, est. printed base + lid + arm
bob_mass   = 56;    // g = 20 x M6 carrosseriering @ 2.79 g

/* ---------- SERVO (P1S: 30 x 24 x 12, 270 deg, 3 kg.cm) ------------ */
servo_l   = 30;
servo_w   = 24;
servo_h   = 12;
shaft_off = 9.5;    // shaft axis from the -X end of the body. MEASURE.
shaft_d   = 5.0;
horn_od   = 16.5;
horn_th   = 2.0;
horn_bolt_r = 6.0;  // MEASURE your metal horn
horn_bolt_n = 4;

/* ---------- PIVOT / SWEEP ------------------------------------------ */
pivot_x     = -12;  // on the centreline: clears all 4 mounting bosses
pivot_y     = 0;
sweep_start = 0;    // 0 = diagonal, -45 = lateral, +45 = fore/aft
sweep_deg   = 90;   // 30 is too little. 90..180 is the useful range.
arm_len     = 34;   // pivot -> bob centre. Slidable, so tunable.

/* ---------- ROD / ARM ---------------------------------------------- */
rod_size = 6;
rod_bore = 6.6;
boss_r0  = 11;
boss_len = 14;
boss_od  = 11;
z_rod    = 8;
hub_d    = 18;
hub_h    = 5;
web_w    = 10;
web_t    = 3;

/* ---------- ENCLOSURE ---------------------------------------------- */
base_th  = 3.5;     // thicker than v2 so the counterbores have a floor
wall     = 2.4;
lid_th   = 4.0;
stud_d   = 3.4;
nut_af   = 6.2;
nut_h    = 2.6;
stud_pad = 9;
cable_w  = 6;
clr      = 0.4;

$fn = 64;

/* ========================= DERIVED ================================= */
plate_len = hole_dx + 2*plate_edge;
plate_wid = hole_dy + 2*plate_edge;

chord   = 2*arm_len*sin(sweep_deg/2);
rod_m   = 0.17*(arm_len+25)/1000*1000;
M_total = robot_mass + base_mass + bob_mass + rod_m;
dx_com  = bob_mass*chord/M_total;

cav_x0  = -shaft_off;
cav_x1  =  servo_l - shaft_off;
comp_x0 = cav_x0 - wall;
comp_x1 = cav_x1 + wall + stud_pad;
comp_y0 = -servo_w/2 - wall;
comp_y1 =  servo_w/2 + wall;
comp_z1 = base_th + servo_h;
lid_z0  = comp_z1;
arm_z0  = lid_z0 + lid_th + horn_th;
boss_r1 = boss_r0 + boss_len;

stud_x  = cav_x1 + wall + stud_pad/2;
stud_y  = 8;

echo(str(">>> plate ", plate_len, " x ", plate_wid, " mm, screws at +-",
         hole_dx/2, " / +-", hole_dy/2));
echo(str(">>> screw length needed >= ", standoff_h + base_th - cb_h,
         " mm of shank ABOVE the board, plus thread engagement"));
echo(str(">>> arm base sits ", arm_z0, " mm above the plate, ",
         arm_z0 + standoff_h, " mm above the board"));
echo(str(">>> sweep ", sweep_deg, " deg, L=", arm_len,
         " -> chord ", chord, " mm"));
echo(str(">>> CoM shift = ", dx_com, " mm  (M_total ", M_total, " g)"));

// which axis does the payload shift along?
module axis_report(){
   let(a0 = sweep_start, a1 = sweep_start + sweep_deg,
       x0 = pivot_x + arm_len*cos(a0), y0 = pivot_y + arm_len*sin(a0),
       x1 = pivot_x + arm_len*cos(a1), y1 = pivot_y + arm_len*sin(a1))
      echo(str("    bob (", x0, ",", y0, ") -> (", x1, ",", y1,
               ")   dx=", x1-x0, "  dy=", y1-y0));
}
echo(">>> payload travel:");
axis_report();

// does the compartment footprint clash with a mounting boss?
module boss_check(){
   for(sx=[-1,1]) for(sy=[-1,1])
      let(hx = sx*hole_dx/2, hy = sy*hole_dy/2,
          hit = (pivot_x+comp_x0-foot_od/2 < hx) && (hx < pivot_x+comp_x1+foot_od/2)
             && (pivot_y+comp_y0-foot_od/2 < hy) && (hy < pivot_y+comp_y1+foot_od/2))
      if(hit) echo(str("!!! compartment clashes with mounting boss at (",
                       hx, ",", hy, ") - move pivot_x / pivot_y"));
}
boss_check();

/* ========================= MOUNTING PLATE ==========================
   Drop-in replacement for the strap plate in the slider design too:
   just swap plate() for this and keep everything above base_th.        */
// one standoff spacer: prints on end, no supports, perfect bed contact
module spacer(){
   difference(){
      cylinder(d=foot_od, h=standoff_h);
      translate([0,0,-1]) cylinder(d=screw_d, h=standoff_h+2);
   }
}

module mount_plate(){
   difference(){
      union(){
         translate([-plate_len/2, -plate_wid/2, 0])
            cube([plate_len, plate_wid, base_th]);
         // integral feet only if explicitly asked for (see note above)
         if(integral_feet)
            for(sx=[-1,1]) for(sy=[-1,1])
               translate([sx*hole_dx/2, sy*hole_dy/2, -standoff_h])
                  cylinder(d=foot_od, h=standoff_h+0.01);
      }
      for(sx=[-1,1]) for(sy=[-1,1])
         translate([sx*hole_dx/2, sy*hole_dy/2, 0]){
            translate([0,0,-standoff_h-1])
               cylinder(d=screw_d, h=standoff_h+base_th+2);   // through hole
            translate([0,0,base_th-cb_h])
               cylinder(d=cb_d, h=cb_h+0.1);                  // flush head
         }
      // optional relief so the plate underside clears low components.
      // 0 = off (best first layer, no bridging). Redundant if your
      // spacers are already tall enough to clear everything.
      if(underside_relief > 0)
         translate([0,0,-1]) linear_extrude(underside_relief+1)
            square([hole_dx-foot_od-2, hole_dy-foot_od-2], center=true);
   }
}

/* ========================= COMPARTMENT ============================= */
// local frame: shaft axis at origin, servo body along +X
module compartment(){
   difference(){
      translate([comp_x0, comp_y0, 0])
         cube([comp_x1-comp_x0, comp_y1-comp_y0, comp_z1]);
      translate([cav_x0-clr, -servo_w/2-clr, base_th])
         cube([servo_l+2*clr, servo_w+2*clr, servo_h+1]);      // servo cavity
      translate([cav_x1-1, -cable_w/2, base_th+servo_h-cable_w])
         cube([wall+2, cable_w, cable_w+1]);                   // cable exit
      for(sy=[-1,1]) translate([stud_x, sy*stud_y, -1])
         cylinder(d=stud_d, h=comp_z1+2);                      // lid studs
   }
}

module base(){
   mount_plate();
   translate([pivot_x, pivot_y, 0]) compartment();
}

/* ========================= LID ===================================== */
module lid(){
   difference(){
      translate([comp_x0, comp_y0, 0])
         cube([comp_x1-comp_x0, comp_y1-comp_y0, lid_th]);
      translate([0,0,-1]) cylinder(d=horn_od+2, h=lid_th+2);
      for(sy=[-1,1]){
         translate([stud_x, sy*stud_y, -1]) cylinder(d=stud_d, h=lid_th+2);
         translate([stud_x, sy*stud_y, lid_th-nut_h])
            cylinder(d=nut_af/cos(30), h=nut_h+0.1, $fn=6);
      }
   }
}

/* ========================= ARM ===================================== */
module arm(){
   difference(){
      union(){
         cylinder(d=hub_d, h=hub_h);
         translate([0, -web_w/2, 0]) cube([boss_r1, web_w, web_t]);
         translate([boss_r0, -web_w/2, web_t-0.01])
            cube([boss_len, web_w, z_rod-web_t+0.01]);
         translate([boss_r0, 0, z_rod]) rotate([0,90,0])
            cylinder(d=boss_od, h=boss_len);
      }
      translate([0,0,-1]) cylinder(d=shaft_d+0.6, h=hub_h+2);
      translate([0,0,-0.01]) cylinder(d=horn_od+0.5, h=horn_th+0.3);
      for(k=[0:horn_bolt_n-1]) rotate(k*360/horn_bolt_n)
         translate([horn_bolt_r,0,-1]) cylinder(d=2.2, h=hub_h+2);
      translate([boss_r0-1, 0, z_rod]) rotate([0,90,0])
         cylinder(d=rod_bore, h=boss_len+2);
   }
}

/* ========================= LAYOUT ================================== */
module bob(){
   // 20 x M6 carrosseriering (18mm OD x 1.6mm) = 56 g, 32 mm stack
   rotate([0,90,0]) cylinder(d=18, h=32, center=true);
}

module layout(){
   color("lightsteelblue") mount_plate();
   translate([pivot_x, pivot_y, 0]){
      color("gainsboro") compartment();
      color("silver") translate([0,0,lid_z0]) lid();
      rotate([0,0,sweep_start]) translate([0,0,arm_z0]){
         color("khaki") arm();
         color("goldenrod") translate([boss_r0+2, 0, z_rod])
            rotate([0,90,0]) cylinder(d=rod_size, h=arm_len+22);
         color("dimgray") translate([arm_len, 0, z_rod]) bob();
      }
      rotate([0,0,sweep_start+sweep_deg]) translate([0,0,arm_z0])
         color([0.9,0.7,0.2,0.25]){
            translate([boss_r0+2, 0, z_rod]) rotate([0,90,0])
               cylinder(d=rod_size, h=arm_len+22);
            translate([arm_len, 0, z_rod]) bob();
         }
   }
}

if      (part=="layout") layout();
else if (part=="base")   base();
else if (part=="lid")    lid();
else if (part=="arm")    arm();
else if (part=="spacer") spacer();

/* =====================================================================
   MOUNTING NOTES
     - You need LONGER screws than the stock ones: they must pass through
       the feet and plate (standoff_h + base_th - cb_h of extra shank)
       and still engage the original posts. Check thread pitch matches.
     - If the stock screws are captive/short, an alternative is to leave
       them alone and use 4 nylon standoffs of the same pitch.
     - standoff_h is the single biggest CoM-height penalty in this design.
       Measure the tallest obstruction and trim it to just clear.

   PRINTING (Bambu, PLA)
     base : plate-down, 0.20mm, 3-4 walls, 15-20% gyroid. The feet are
            the only bed contact -> add a 4mm brim for adhesion.
     lid  : flat, 0.16mm, 4 walls, ~40% infill.
     arm  : web-down, 0.16mm, 4 walls, 40-60% infill. Run the M6 rod
            through the boss bore to clear it.

   ASSEMBLY ORDER MATTERS
     1. Fit the servo, lid, and M3 studs BEFORE bolting the plate down -
        the lid studs are inboard and awkward to reach afterwards.
     2. Centre the servo in software, then fit the arm at mid-sweep.
     3. Bolt the plate to the board. Then rod, then washer stack.
     4. Sweep by hand through the full range checking LEG CLEARANCE
        before powering it.

   STILL UNVERIFIED (needs the robot in hand)
     - screw_d, standoff_h, shaft_off, horn_bolt_r
     - leg clearance through the sweep: the bob is 18mm across and sits
       ~(arm_z0 + standoff_h) above the board. Set software end stops.
   ===================================================================== */
