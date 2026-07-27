/* =====================================================================
   Bittle CoM-Shift Harness  —  pinion (standalone)
   ---------------------------------------------------------------------
   Involute spur pinion, built as a SOLID HUB + narrowing teeth so it
   extrudes to a closed 2-manifold (fixes the "not closed" error).

   IMPORTANT: to mesh with the rack in the main file, these three MUST
   match the main file exactly:  gear_module, pinion_teeth, pressure_angle
   ===================================================================== */

/* ---- gear (keep identical to the main harness file) ---- */
gear_module    = 1.0;
pinion_teeth   = 17;
pressure_angle = 20;

/* ---- this part ---- */
pinion_face   = 6;      // thickness (Z)
servo_shaft_d = 5;      // central clearance for the servo shaft
horn_bore_d   = 16;     // counterbore to seat a metal round servo horn
horn_bore_h   = 3;      // counterbore depth (from the top face)
horn_bolt_d   = 2.2;    // M2 clearance
horn_bolt_r   = 6;      // bolt-circle radius (measure your horn, then set)
horn_bolt_n   = 4;      // number of horn screws
backlash      = 0.15;   // mm removed per flank (match main file)

$fn = 96;
gsteps = 20;            // involute sampling per flank

/* ---------------- gear math ---------------- */
function polar(r,a) = [r*cos(a), r*sin(a)];
function inv_deg(a) = (tan(a) - a*PI/180) * 180/PI;   // involute fn, deg->deg

module pinion_2d(){
   pr = gear_module*pinion_teeth/2;      // pitch radius
   br = pr*cos(pressure_angle);          // base circle
   ar = pr + gear_module;                // addendum (outer)
   rr = pr - 1.25*gear_module;           // root
   bl_ang   = (backlash/pr)*180/PI;      // backlash as an angle
   half     = 90/pinion_teeth - bl_ang/2;// half-tooth angular width at pitch
   base_ang = half + inv_deg(pressure_angle);
   t_out    = acos(br/ar);
   // one flank, root -> tip, on the + side. angle DECREASES with radius
   // (base_ang - inv_deg(t)) so the tooth NARROWS toward the tip.
   left = concat(
      [ polar(rr, base_ang) ],
      [ for(i=[0:gsteps]) let(t = i/gsteps*t_out, r = br/cos(t))
           polar(r, base_ang - inv_deg(t)) ]
   );
   // mirror for the - side flank, traversed tip -> root
   right = [ for(i=[len(left)-1:-1:0]) [left[i][0], -left[i][1]] ];
   tooth = concat(left, right);          // closed by the root chord

   union(){
      circle(r = rr + 0.02);             // SOLID HUB — teeth attach here
      for(k=[0:pinion_teeth-1]) rotate(k*360/pinion_teeth) polygon(tooth);
   }
}

module pinion(){
   difference(){
      linear_extrude(pinion_face, convexity=10) pinion_2d();
      translate([0,0,-1]) cylinder(d=servo_shaft_d, h=pinion_face+2);       // shaft
      translate([0,0,pinion_face-horn_bore_h])
         cylinder(d=horn_bore_d, h=horn_bore_h+0.1);                        // horn seat
      for(k=[0:horn_bolt_n-1])
         rotate(k*360/horn_bolt_n) translate([horn_bolt_r,0,-1])
            cylinder(d=horn_bolt_d, h=pinion_face+2);                       // horn bolts
   }
}

pinion();

echo(str(">>> pitch dia = ", gear_module*pinion_teeth, " mm  (must match rack module/PA)"));
