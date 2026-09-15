# -*- coding: utf-8 -*-
"""Fold the legs so the schematic is ~20% less tall. Legs, ground, and the
per-leg Delta annotations only -- trunk, payload, straps and the xi triad stay put."""
import re, io, sys

src, dst = sys.argv[1], sys.argv[2]
DY = float(sys.argv[3])            # how far the feet (and ground) rise [cm]
KX, KY = float(sys.argv[4]), float(sys.argv[5])   # knee offset along body +x
s = io.open(src, encoding='utf-8').read()

OLDH, NEWH = 5.15, round(5.15 - DY, 3)

def cm(v): return float(v.replace('cm', ''))
def attrs(t): return dict(re.findall(r'([\w:-]+)="([^"]*)"', t))

# trunk bottom silhouette: HLB->FLB then FLB->FRB (a knee above this is swallowed)
HLB, FLB, FRB = (4.15, 2.42), (6.25, 2.77), (8.00, 2.31)
def silhouette(x):
    if HLB[0] <= x <= FLB[0]:
        return HLB[1] + (x - HLB[0]) / (FLB[0] - HLB[0]) * (FLB[1] - HLB[1])
    if FLB[0] < x <= FRB[0]:
        return FLB[1] + (x - FLB[0]) / (FRB[0] - FLB[0]) * (FRB[1] - FLB[1])
    return -9e9

# ---- 1. legs: keep the hip, raise the foot, fold the knee further out
shank_shift = {}
def fix_leg(m):
    tag = m.group(0); a = attrs(tag)
    x0, y0 = cm(a['svg:x']), cm(a['svg:y'])
    w, h = cm(a['svg:width']), cm(a['svg:height'])
    vb = [float(v) for v in a['svg:viewBox'].split()]
    pts = [tuple(float(v) for v in p.split(',')) for p in a['draw:points'].split()]
    sx, sy = w / vb[2], h / vb[3]
    real = [(x0 + px * sx, y0 + py * sy) for px, py in pts]
    if len(real) != 3:
        return tag
    hip, knee_old, foot = real
    foot_n = (foot[0], foot[1] - DY)
    knee_n = ((hip[0] + foot_n[0]) / 2.0 + KX, (hip[1] + foot_n[1]) / 2.0 + KY)
    lim = silhouette(knee_n[0]) + 0.07              # keep the knee clear of the body
    if knee_n[1] < lim:
        knee_n = (knee_n[0], lim)
    shank_shift[round(hip[0], 2)] = (
        (knee_n[0] + foot_n[0]) / 2.0 - (knee_old[0] + foot[0]) / 2.0,
        (knee_n[1] + foot_n[1]) / 2.0 - (knee_old[1] + foot[1]) / 2.0,
        knee_old, knee_n)
    new = [hip, knee_n, foot_n]
    xs = [p[0] for p in new]; ys = [p[1] for p in new]
    nx, ny = min(xs), min(ys)
    nw = max(max(xs) - nx, 0.001); nh = max(max(ys) - ny, 0.001)
    d = ' '.join('%d,%d' % (round((p[0] - nx) * 1000), round((p[1] - ny) * 1000)) for p in new)
    t = tag
    for k, v in (('svg:x', '%.3fcm' % nx), ('svg:y', '%.3fcm' % ny),
                 ('svg:width', '%.3fcm' % nw), ('svg:height', '%.3fcm' % nh),
                 ('svg:viewBox', '0 0 %d %d' % (int(round(nw * 1000)), int(round(nh * 1000)))),
                 ('draw:points', d)):
        t = re.sub(r'%s="[^"]*"' % k, '%s="%s"' % (k, v), t)
    return t

s = re.sub(r'<draw:polyline\b[^>]*>', fix_leg, s)

# ---- 2. ground plane rises with the feet
def fix_ground(m):
    tag = m.group(0); a = attrs(tag)
    if not (abs(cm(a['svg:x']) - 3.63) < 0.05 and abs(cm(a['svg:width']) - 4.74) < 0.05):
        return tag
    return re.sub(r'svg:y="[^"]*"', 'svg:y="%.3fcm"' % (cm(a['svg:y']) - DY), tag)
s = re.sub(r'<draw:polygon\b[^>]*>', fix_ground, s)

# ---- 3. knee joints follow their leg (hips are untouched)
KNEES = {(5.51, 2.96): 5.89, (7.62, 3.31): 8.00, (5.86, 3.59): 6.25, (3.74, 3.23): 4.15}
def fix_circle(m):
    tag = m.group(0); a = attrs(tag)
    cx = cm(a['svg:x']) + cm(a['svg:width']) / 2; cy = cm(a['svg:y']) + cm(a['svg:height']) / 2
    for (kx, ky), hipx in KNEES.items():
        if abs(cx - kx) < 0.04 and abs(cy - ky) < 0.04:
            _, _, _, kn = shank_shift[round(hipx, 2)]
            t = re.sub(r'svg:x="[^"]*"', 'svg:x="%.3fcm"' % (kn[0] - cm(a['svg:width']) / 2), tag)
            return re.sub(r'svg:y="[^"]*"', 'svg:y="%.3fcm"' % (kn[1] - cm(a['svg:height']) / 2), t)
    return tag
s = re.sub(r'<draw:circle\b[^>]*>', fix_circle, s)

# ---- 4. Delta arrows and labels ride with their own shank
ARROW_LEG = {6.31: 6.25, 4.15: 4.15, 8.08: 8.00, 5.38: 5.89}
def fix_line(m):
    tag = m.group(0); a = attrs(tag)
    x1, y1, y2 = cm(a['svg:x1']), cm(a['svg:y1']), cm(a['svg:y2'])
    if abs(cm(a['svg:x2']) - x1) > 0.02 or abs(y2 - y1) > 0.6:
        return tag                                   # not a Delta stroke
    for ax, hipx in ARROW_LEG.items():
        if abs(x1 - ax) < 0.05:
            dx, dy = shank_shift[round(hipx, 2)][:2]
            t = tag
            for k, v in (('svg:x1', x1 + dx), ('svg:y1', cm(a['svg:y1']) + dy),
                         ('svg:x2', cm(a['svg:x2']) + dx), ('svg:y2', cm(a['svg:y2']) + dy)):
                t = re.sub(r'%s="[^"]*"' % k, '%s="%.3fcm"' % (k, v), t)
            return t
    return tag
s = re.sub(r'<draw:line\b[^>]*>', fix_line, s)

LABEL_LEG = {u'\u03941': 6.25, u'\u03943': 4.15, u'\u03942': 8.00, u'\u03944': 5.89}
def fix_frame(m):
    tag = m.group(0); a = attrs(tag[:400])
    if 'svg:x' not in a:
        return tag
    txt = re.sub(r'<[^>]*>', '', tag).strip()
    fx = cm(a['svg:x'])
    for lbl, hipx in LABEL_LEG.items():
        if txt == lbl:
            dx, dy = shank_shift[round(hipx, 2)][:2]
            t = re.sub(r'svg:x="[^"]*"', 'svg:x="%.3fcm"' % (fx + dx), tag)
            return re.sub(r'svg:y="[^"]*"', 'svg:y="%.3fcm"' % (cm(a['svg:y']) + dy), t)
    return tag
s = re.sub(r'<draw:frame\b.*?</draw:frame>', fix_frame, s, flags=re.S)

# ---- 5. shrink the page by exactly what the legs gave up
s = s.replace('fo:page-height="5.15cm"', 'fo:page-height="%.3fcm"' % NEWH)
io.open(dst, 'w', encoding='utf-8').write(s)
for hx, (dx, dy, ko, kn) in sorted(shank_shift.items()):
    print('  hip x=%.2f  knee %s -> (%.2f,%.2f)   shank shift (%.2f,%.2f)'
          % (hx, '(%.2f,%.2f)' % ko, kn[0], kn[1], dx, dy))
print('page height %.2f -> %.2f cm (%.0f%% shorter)' % (OLDH, NEWH, 100 * DY / OLDH))
