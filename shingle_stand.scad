// Shingle Scanner camera stand v1 (parametric, NOT yet test-printed)
// Matches shingle_scanner_print_pack.pdf, pages 6-8.
// Render one part at a time:  part = "foot" | "post_1" | "post_2" | "post_3" | "carriage" | "all"
// Units: mm. z = 0 is the baseboard surface.

part = "all";

// ---------- standard (keep in sync with the PDF / software) ----------
tray_thickness = 5;      // tray board + strip; engraved heights are above the MAT
pin_to_plate   = 25;     // carriage: pin centre -> plate top (phone back)
h_min = 230; h_max = 400; h_step = 10;   // engraved lens heights
arm_reach      = 160;    // post centre -> lens window centre
foot_front     = 40;     // post centre -> foot front edge (= strip top line)

// ---------- mechanics ----------
post_w    = 30;
clear     = 0.3;         // per-side sliding clearance
pin_d     = 5.4;         // 5 mm dowel / M5 bolt
joints    = [135, 275];  // segment joint heights
post_top  = 415;
peg_w     = 14;  peg_len = 20;  peg_clear = 0.2;
sleeve_wall = 6; sleeve_h = 2 * pin_to_plate;
plate = 120; plate_t = 4; window = 70;
arm_w = 30; arm_t = 12;
label_size = 4.5;
$fn = 40;

function pin_z(h) = h - pin_to_plate + tray_thickness;
seg_bounds = [0, joints[0], joints[1], post_top];

// ---------------- post ----------------
module pin_holes() {
    for (h = [h_min : h_step : h_max])
        translate([0, 0, pin_z(h)]) rotate([0, 90, 0])
            cylinder(d = pin_d, h = post_w + 40, center = true);
}

module labels(z0, z1) {
    for (h = [h_min : h_step : h_max]) {
        z = pin_z(h);
        if (z > z0 + 3 && z < z1 - 3)
            translate([post_w / 2 - 0.8, 4, z - label_size / 2])
                rotate([90, 0, 90]) linear_extrude(2)
                    text(str(h), size = label_size, font = "Liberation Sans:style=Bold");
    }
}

module post_segment(i) {            // i = 1..3
    z0 = seg_bounds[i - 1]; z1 = seg_bounds[i];
    difference() {
        union() {
            translate([-post_w / 2, -post_w / 2, z0]) cube([post_w, post_w, z1 - z0]);
            if (i < 3)   // peg on top
                translate([-(peg_w - peg_clear) / 2, -(peg_w - peg_clear) / 2, z1 - 0.01])
                    cube([peg_w - peg_clear, peg_w - peg_clear, peg_len]);
        }
        if (i > 1)       // socket at bottom
            translate([-(peg_w + peg_clear) / 2, -(peg_w + peg_clear) / 2, z0 - 0.01])
                cube([peg_w + peg_clear, peg_w + peg_clear, peg_len + 0.6]);
        pin_holes();     // also drills the pegs, so a pin can lock a joint
        labels(z0, z1);
    }
}

// ---------------- foot ----------------
module foot() {
    boss = post_w + 2 * clear + 2 * 7;
    difference() {
        union() {
            translate([-100, foot_front - 100, 0]) cube([200, 100, 8]);
            translate([-boss / 2, -boss / 2, 0]) cube([boss, boss, 30]);
        }
        // post socket, through: post stands on the baseboard
        translate([-(post_w / 2 + clear), -(post_w / 2 + clear), -1])
            cube([post_w + 2 * clear, post_w + 2 * clear, 40]);
        // centre notch on the front edge (tray alignment)
        translate([0, foot_front, -1]) rotate([0, 0, 45]) cube([4, 4, 12], center = false);
        // screw holes, countersunk
        for (x = [-85, 85], y = [foot_front - 85, foot_front - 15])
            translate([x, y, -1]) {
                cylinder(d = 5, h = 12);
                translate([0, 0, 5.5]) cylinder(d1 = 5, d2 = 10, h = 2.6);
            }
    }
}

// ---------------- carriage (sleeve + arm + phone plate) ----------------
module carriage() {
    inner = post_w + 2 * clear;
    outer = inner + 2 * sleeve_wall;
    top = sleeve_h;                           // plate top = pin + pin_to_plate
    difference() {
        union() {
            translate([-outer / 2, -outer / 2, 0]) cube([outer, outer, sleeve_h]);
            // arm
            translate([-arm_w / 2, outer / 2 - 1, top - arm_t])
                cube([arm_w, arm_reach - plate / 2 - outer / 2 + 2, arm_t]);
            // plate
            translate([-plate / 2, arm_reach - plate / 2, top - plate_t]) cube([plate, plate, plate_t]);
            // gusset under the arm
            translate([-4, outer / 2 - 1, 0]) rotate([90, 0, 90]) linear_extrude(8)
                polygon([[0, 0], [0, top - arm_t], [70, top - arm_t]]);
        }
        translate([-inner / 2, -inner / 2, -1]) cube([inner, inner, sleeve_h + 2]);
        translate([0, 0, pin_to_plate]) rotate([0, 90, 0]) cylinder(d = pin_d, h = outer + 2, center = true);
        // lens window
        translate([-window / 2, arm_reach - window / 2, top - plate_t - 1]) cube([window, window, plate_t + 2]);
        // crosshair notches at the window edges (align the MAIN lens with their intersection)
        for (a = [0, 90, 180, 270])
            translate([0, arm_reach, top - 0.8]) rotate([0, 0, a]) translate([window / 2, -0.6, 0])
                cube([12, 1.2, 1]);
        // rubber-band hooks on the three free edges
        for (x = [-plate / 2, plate / 2 - 4])
            for (y = [arm_reach - 30, arm_reach + 30])
                translate([x, y - 5, top - plate_t - 1]) cube([4, 10, plate_t + 2]);
        translate([-5, arm_reach + plate / 2 - 4, top - plate_t - 1]) cube([10, 4, plate_t + 2]);
    }
}

// ---------------- output ----------------
if (part == "foot") foot();
else if (part == "post_1") post_segment(1);
else if (part == "post_2") translate([0, 0, -joints[0]]) post_segment(2);
else if (part == "post_3") translate([0, 0, -joints[1]]) post_segment(3);
else if (part == "carriage")   // printed plate-side down
    rotate([180, 0, 0]) translate([0, 0, -sleeve_h]) carriage();
else {                          // assembly preview at h = 360 (C6, 26 mm lens)
    color("silver") foot();
    for (i = [1 : 3]) color(i % 2 ? "lightblue" : "lightsteelblue") post_segment(i);
    color("orange") translate([0, 0, pin_z(360) - pin_to_plate]) carriage();
    color("wheat", 0.4) translate([-300, foot_front, -1]) cube([600, 310, 1]);
}
