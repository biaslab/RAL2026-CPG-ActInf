/* =====================================================================
   Bittle CoM-Shift Harness  —  parametric v0
   Single Petoi P1S servo · rack-and-pinion linear slider · diagonal mount
   ---------------------------------------------------------------------
   Units: millimetres, degrees.  Print PLA/PETG for v0.
   Use a STEEL guide rod (rod_dia) rather than printing the sliding
   surface — printed-on-printed prismatic joints bind badly.

   TWO THINGS TO VERIFY ON THE BENCH (not trustworthy from the render):
     1. Pinion<->rack CENTRE DISTANCE. Tune with 'backlash'/'center_extra'.
     2. Z registration of pinion face vs rack face ('mesh_z').
   Everything else is dimensionally driven by the parameters below.
   ===================================================================== */

/* ---------- PART SELECTOR ------------------------------------------ */
// "layout" = assembled visualisation; the rest export a single flat part
part = "carriage";   // ["layout","base","carriage","pinion","rack"]

/* ---------- ROBOT / DECK ------------------------------------------- */
deck_len   = 100;  // electronics deck, long edge  (measure next week)
deck_wid   = 50;   // electronics deck, short edge
robot_mass = 290;  // g, bare Bittle
mech_mass  = 45;   // g, est. servo + harness + rack (refine after weighing)
m_shift    = 100;  // g, swappable payload slug  <-- your independent variable

/* ---------- SERVO (Petoi P1S: 30x24x12, 270 deg, 3 kg.cm) ---------- */
servo_l = 30; servo_w = 24; servo_h = 12;  // body envelope
servo_travel  = 270;                        // usable degrees
servo_shaft_d = 5;                          // shaft / horn-boss clearance

/* ---------- GEAR --------------------------------------------------- */
gear_module    = 1.0;   // >=1 so printed teeth survive
pinion_teeth   = 17;    // pitch dia = module * teeth
pressure_angle = 20;
pinion_face    = 6;     // pinion thickness  (Z)
rack_face      = 8;     // rack tooth height (Z)
backlash       = 0.15;  // mm, thinned per flank
center_extra   = 0.0;   // mm, add to centre distance to loosen mesh

/* ---------- GUIDE / CARRIAGE --------------------------------------- */
rod_dia       = 3.0;    // steel guide rods
rod_spacing   = 22;     // centre-to-centre (anti-rotation)
carriage_len  = 30;
mass_pocket_d = 21;     // shallow well for stacked ~20mm washers / M6 nuts
mass_pocket_h = 12;     // keep shallow: taller = higher CoM = tippier
clr           = 0.4;    // general slip-fit clearance

/* ---------- BASE --------------------------------------------------- */
base_th      = 2.5;
strap_slot_w = 4;       // velcro / zip-tie slots to strap over the deck
wall         = 2.4;

$fn = 48;

/* ========================= DERIVED ================================= */
pitch_r  = gear_module * pinion_teeth / 2;
stroke   = pitch_r * servo_travel * PI / 180;      // rack travel for full sweep
M_total  = robot_mass + mech_mass + m_shift;
dx_com   = m_shift * stroke / M_total;             // 1-D CoM shift magnitude
diag_ang = atan2(deck_wid, deck_len);              // diagonal orientation
rack_len = stroke + 6 * gear_module;               // keep meshed across travel

// carriage cross-width (Y): must clear both rods AND the mass well
cw = max(rod_spacing + 2*rod_dia + 2*wall, mass_pocket_d + 2*wall + 2);

// heights
rod_z   = base_th + rod_dia/2 + wall;              // rod centreline above plate
mesh_z  = base_th + servo_h;                       // servo lies flat, shaft up
y_rack  = cw/2 + wall;                             // rack pitch-line Y (local, at wall face)
y_pin   = y_rack + pitch_r + center_extra;         // pinion centre Y (local)
RL      = (stroke + carriage_len)/2 + 8;           // rail half-length

echo(str(">>> pinion pitch dia = ", 2*pitch_r, " mm"));
echo(str(">>> usable stroke    = ", stroke, " mm  over ", servo_travel, " deg"));
echo(str(">>> diagonal angle   = ", diag_ang, " deg (fore/aft:lateral set by deck)"));
echo(str(">>> total mass       = ", M_total, " g"));
echo(str(">>> predicted CoM shift = ", dx_com, " mm  @ ", m_shift, " g payload"));

/* ========================= GEAR MATH =============================== */
function polar(r,a) = [r*cos(a), r*sin(a)];
function rot2(p,a)  = [p[0]*cos(a)-p[1]*sin(a), p[0]*sin(a)+p[1]*cos(a)];
function inv_deg(a) = (tan(a) - a*PI/180) * 180/PI;   // involute fn, deg->deg

gsteps = 14;
function flank(br,ar) =
   let(t_out = acos(br/ar))
   [ for(i=[0:gsteps]) let(t = i/gsteps*t_out, r = br/cos(t))
        polar(r, inv_deg(t)) ];

// involute spur pinion, centred on origin, in XY.
// Built as SOLID HUB + narrowing teeth so it extrudes to a closed manifold.
module pinion_2d(){
   pr = pitch_r;
   br = pr*cos(pressure_angle);          // base circle
   ar = pr + gear_module;                // addendum (outer) radius
   rr = pr - 1.25*gear_module;           // root radius
   bl_ang   = (backlash/pr)*180/PI;      // backlash expressed as angle
   half     = 90/pinion_teeth - bl_ang/2;// half-tooth angular width at pitch
   base_ang = half + inv_deg(pressure_angle);
   t_out    = acos(br/ar);
   // flank root->tip, +side; angle DECREASES with radius so teeth narrow
   left  = concat([ polar(rr, base_ang) ],
                  [ for(i=[0:gsteps]) let(t=i/gsteps*t_out, r=br/cos(t))
                       polar(r, base_ang - inv_deg(t)) ]);
   right = [ for(i=[len(left)-1:-1:0]) [left[i][0], -left[i][1]] ]; // tip->root, mirrored
   tooth = concat(left, right);
   union(){
      circle(r = rr + 0.02);             // solid hub the teeth attach to
      for(k=[0:pinion_teeth-1]) rotate(k*360/pinion_teeth) polygon(tooth);
   }
}

module pinion(){
   difference(){
      linear_extrude(pinion_face) pinion_2d();
      translate([0,0,-1]) cylinder(d=servo_shaft_d+clr, h=pinion_face+2); // shaft clr
      // counterbore to seat a metal servo horn (bolt the pinion to it)
      translate([0,0,pinion_face-2.5]) cylinder(d=16, h=3);
      // 4x M2 bolt circle for the horn
      for(k=[0:3]) rotate(k*90) translate([6,0,-1]) cylinder(d=2.2, h=pinion_face+2);
   }
}

// straight-flank rack (exact conjugate of the involute pinion), teeth +Y
module rack_2d(len){
   p   = PI*gear_module;                 // circular pitch
   n   = floor(len/p);
   add = gear_module;                    // addendum (up)
   ded = 1.25*gear_module;               // dedendum (down)
   ht  = p/4 - backlash/2;               // half tooth thickness at pitch line
   base_h = 2;                           // solid backing below root
   union(){
      translate([0, -(ded+base_h)]) square([n*p, base_h]);          // backing strip
      for(i=[0:n-1]) let(cx=(i+0.5)*p)
         polygon([[cx-(ht+ded*tan(pressure_angle)), -ded],
                  [cx-(ht-add*tan(pressure_angle)),  add],
                  [cx+(ht-add*tan(pressure_angle)),  add],
                  [cx+(ht+ded*tan(pressure_angle)), -ded]]);
   }
}
module rack(){ linear_extrude(rack_face) rack_2d(rack_len); }

/* ========================= CARRIAGE ================================ */
// slides on the two rods (X), carries the rack wall (+Y) and the mass well
module carriage(){
   ch = rod_dia + 2*wall;                        // body height around rods
   wall_h = mesh_z + rack_face;                  // rack wall rises to mesh plane
   difference(){
      union(){
         // body block straddling the rods
         translate([-carriage_len/2, -cw/2, base_th]) cube([carriage_len, cw, ch]);
         // tall +Y wall that carries the rack up at the mesh plane
         translate([-rack_len/2, cw/2-0.01, base_th])
            cube([rack_len, wall, wall_h-base_th]);
         // mass well (cup) centred on the carriage
         translate([0,0,base_th])
            cylinder(d=mass_pocket_d+2*wall, h=ch+mass_pocket_h);
      }
      // rod bores (slip fit)
      for(sy=[-1,1])
         translate([-carriage_len/2-1, sy*rod_spacing/2, base_th+ch/2])
            rotate([0,90,0]) cylinder(d=rod_dia+clr, h=carriage_len+2);
      // mass pocket cavity, open from top
      translate([0,0,base_th+ch]) cylinder(d=mass_pocket_d, h=mass_pocket_h+1);
   }
   // rack teeth on the outer face of the +Y wall, at the mesh plane
   translate([-rack_len/2, cw/2+wall-0.01, mesh_z]) rack();
}

/* ========================= BASE ==================================== */
// deck-shaped plate with strap slots + rail furniture on the diagonal
module plate(){
   difference(){
      translate([-deck_len/2,-deck_wid/2,0]) cube([deck_len,deck_wid,base_th]);
      // strap slots near the four edges
      for(sx=[-1,1]) translate([sx*(deck_len/2-8), 0, -1])
         cube([strap_slot_w, deck_wid*0.6, base_th+2], center=true);
   }
}

// rail furniture in a LOCAL frame (rail along +X); rotated onto plate below
module rail_furniture(show_moving){
   // end bosses holding the rod ends
   for(sx=[-1,1])
      translate([sx*RL, 0, base_th]) difference(){
         translate([-3, -(rod_spacing/2+rod_dia+wall), 0])
            cube([6, rod_spacing+2*rod_dia+2*wall, rod_dia+2*wall]);
         for(sy=[-1,1])
            translate([-4, sy*rod_spacing/2, (rod_dia+2*wall)/2])
               rotate([0,90,0]) cylinder(d=rod_dia+0.1, h=8);   // rod press/glue
      }
   // servo cradle at travel centre (pinion sits at x=0 so rack stays meshed)
   translate([0, y_pin, base_th]) difference(){
      translate([-(servo_l/2+wall), -(servo_w/2+wall), 0])
         cube([servo_l+2*wall, servo_w+2*wall, servo_h+wall]);
      translate([-servo_l/2, -servo_w/2, wall]) cube([servo_l, servo_w, servo_h+1]);
      translate([0,0,-1]) cylinder(d=servo_shaft_d+2, h=servo_h+wall+2);   // shaft slot
   }
   if(show_moving){
      // guide rods (visual)
      color("silver") for(sy=[-1,1])
         translate([-RL, sy*rod_spacing/2, rod_z]) rotate([0,90,0])
            cylinder(d=rod_dia, h=2*RL);
      // pinion on the shaft
      color("tomato") translate([0, y_pin, mesh_z]) pinion();
   }
}

module base(){
   plate();
   rotate([0,0,diag_ang]) rail_furniture(false);
}

/* ========================= ASSEMBLY / EXPORT ======================= */
module layout(){
   color("lightsteelblue") plate();
   rotate([0,0,diag_ang]){
      rail_furniture(true);
      // carriage shown at travel centre, with a translucent slug
      color("khaki")   carriage();
      color([1,0.9,0.4,0.4]) translate([0,0,mesh_z])
         cylinder(d=mass_pocket_d-1, h=mass_pocket_h-2);
   }
}

if      (part=="layout")   layout();
else if (part=="base")     base();
else if (part=="carriage") carriage();
else if (part=="pinion")   pinion();
else if (part=="rack")     rack();

/* =====================================================================
   BOM (v0)
     - 1x Petoi P1S servo  + metal round horn (bolt pinion to it)
     - 2x steel rod, dia = rod_dia, length ~ 2*RL  (cut to fit bosses)
     - swappable slug: stacked steel washers / M6 nuts / tungsten
     - 4x M2 screws (pinion->horn), velcro or zip-ties (strap slots)
   PRINT NOTES
     - pinion & rack: 100% infill, fine layers; module>=1 for tooth strength
     - the tall +Y rack wall is thin — add a gusset if it flexes
     - firmware: NyBoard's PCA9685 has spare channels beyond Bittle's 9;
       wire this servo to a free channel and index it as a 10th joint
   ===================================================================== */
