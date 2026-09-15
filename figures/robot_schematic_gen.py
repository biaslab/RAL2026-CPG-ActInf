# -*- coding: utf-8 -*-
"""robot_schematic.fodp -- CPG traces + VMC posture correction, editable in Impress."""
import math, io, sys
import numpy as np

W, H = 9.0, 5.15
shapes, extra_styles = [], {}

# ---------------------------------------------------------------- primitives
def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

def line(x1, y1, x2, y2, style):
    shapes.append('<draw:line draw:style-name="%s" svg:x1="%.3fcm" svg:y1="%.3fcm" '
                  'svg:x2="%.3fcm" svg:y2="%.3fcm"><text:p/></draw:line>'
                  % (style, x1, y1, x2, y2))

def poly(pts, style, closed=True):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, y0 = min(xs), min(ys)
    w = max(max(xs) - x0, 0.001); h = max(max(ys) - y0, 0.001)
    d = ' '.join('%d,%d' % (round((p[0] - x0) * 1000), round((p[1] - y0) * 1000)) for p in pts)
    tag = 'polygon' if closed else 'polyline'
    shapes.append('<draw:%s draw:style-name="%s" svg:x="%.3fcm" svg:y="%.3fcm" '
                  'svg:width="%.3fcm" svg:height="%.3fcm" svg:viewBox="0 0 %d %d" '
                  'draw:points="%s"><text:p/></draw:%s>'
                  % (tag, style, x0, y0, w, h, int(round(w * 1000)), int(round(h * 1000)),
                     d, tag))

def circle(cx, cy, r, style):
    shapes.append('<draw:ellipse draw:style-name="%s" svg:x="%.3fcm" svg:y="%.3fcm" '
                  'svg:width="%.3fcm" svg:height="%.3fcm"><text:p/></draw:ellipse>'
                  % (style, cx - r, cy - r, 2 * r, 2 * r))

def arc(cx, cy, r, u1, u2, a0, a1, style, n=26):
    pts = [(cx + r * (math.cos(math.radians(a)) * u1[0] + math.sin(math.radians(a)) * u2[0]),
            cy + r * (math.cos(math.radians(a)) * u1[1] + math.sin(math.radians(a)) * u2[1]))
           for a in [a0 + (a1 - a0) * k / n for k in range(n + 1)]]
    poly(pts, style, closed=False)

KIND = {'var': 'Tvar', 'rm': 'Trm', 'sub': 'Tsub', 'sup': 'Tsup',
        'subrm': 'Tsubrm', 'suprm': 'Tsuprm'}

def _frame(cx, cy, runs, w, h, gst):
    body = ''.join('<text:span text:style-name="%s">%s</text:span>' % (KIND[k], esc(t))
                   for t, k in runs)
    shapes.append('<draw:frame draw:style-name="%s" svg:x="%.3fcm" svg:y="%.3fcm" '
                  'svg:width="%.3fcm" svg:height="%.3fcm">'
                  '<draw:text-box><text:p text:style-name="Plbl">%s</text:p></draw:text-box>'
                  '</draw:frame>' % (gst, cx - w / 2, cy - h / 2, w, h, body))

def label(cx, cy, runs, w=0.95, h=0.34):   _frame(cx, cy, runs, w, h, 'gText')
def labelbg(cx, cy, runs, w=0.95, h=0.34): _frame(cx, cy, runs, w, h, 'gTextBg')

def norm(v):
    m = math.hypot(*v); return (v[0] / m, v[1] / m)

def mix(c0, c1, t):
    a = tuple(int(c0[i:i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(c1[i:i + 2], 16) for i in (1, 3, 5))
    return '#%02x%02x%02x' % tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))

def fading_trace(pts, bands=14, c_old='#e3ebf2', c_new='#12496f', width=0.030):
    """Polyline whose older segments fade toward the background."""
    n = len(pts)
    for k in range(bands):
        i0 = int(round(k * (n - 1) / float(bands)))
        i1 = int(round((k + 1) * (n - 1) / float(bands)))
        if i1 <= i0:
            continue
        col = mix(c_old, c_new, (k + 0.5) / bands)
        nm = 'gTr%s' % col[1:]
        extra_styles[nm] = ('draw:fill="none" draw:stroke="solid" svg:stroke-width="%.3fcm" '
                            'svg:stroke-color="%s" svg:stroke-linejoin="round" '
                            'svg:stroke-linecap="round"' % (width, col))
        poly(pts[i0:i1 + 1], nm, closed=False)

# ---------------------------------------------------------------- trace data
TR = np.load(sys.argv[2])                       # rows: x, y, hip, knee
SEG, STEP = 312, 3                              # ~2 gait cycles, subsampled
X, Y, HIPA, KNEEA = [r[-SEG::STEP] for r in TR]

def fit(vals, lo, hi, pad=0.10, flip=False):
    v0, v1 = float(np.min(vals)), float(np.max(vals))
    m = (v1 - v0) * pad or 1e-6
    v0, v1 = v0 - m, v1 + m
    t = (np.asarray(vals) - v0) / (v1 - v0)
    return (hi - (hi - lo) * t) if flip else (lo + (hi - lo) * t)

# ---------------------------------------------------------------- plots
PX0, PX1 = 1.22, 2.60
TS = [PX0 + 0.04 + (PX1 - 0.06 - PX0 - 0.04) * k / (len(X) - 1.0) for k in range(len(X))]

def swing_bands(ytop, ybot):
    """Light band over every interval the oscillator spends in swing (y_i > 0)."""
    k = 0
    while k < len(Y):
        if Y[k] > 0:
            j = k
            while j + 1 < len(Y) and Y[j + 1] > 0:
                j += 1
            if j > k:
                poly([(TS[k], ytop), (TS[j], ytop), (TS[j], ybot), (TS[k], ybot)], 'gSwing')
            k = j + 1
        else:
            k += 1

def signal(vals, ytop, ybot):
    pts = list(zip(TS, fit(vals, ytop, ybot, flip=True)))
    fading_trace(pts, width=0.026)
    circle(pts[-1][0], pts[-1][1], 0.068, 'gHead')

def taxis(ay, tpos):
    line(PX0, ay, PX1 + 0.22, ay, 'gAxisArr')
    label(tpos[0], tpos[1], [('t', 'var')], w=0.34)

# (a) oscillator state
swing_bands(0.42, 1.70)
signal(X, 0.48, 1.00); label(0.66, 0.74, [('x', 'var'), ('i', 'sub')], w=0.92)
signal(Y, 1.12, 1.64); label(0.66, 1.38, [('y', 'var'), ('i', 'sub')], w=0.92)
taxis(1.82, (2.74, 2.00))

line(1.85, 1.98, 1.85, 2.50, 'gArrow')                     # (a) -> (b)

# (b) joint angles of one leg
swing_bands(2.60, 3.88)
signal(HIPA, 2.66, 3.18); label(0.66, 2.92, [(u'\u03c6', 'var'), ('hip', 'suprm')], w=0.92)
signal(KNEEA, 3.30, 3.82); label(0.66, 3.56, [(u'\u03c6', 'var'), ('knee', 'suprm')], w=0.96)
taxis(4.00, (2.74, 4.18))

# ---------------------------------------------------------------- robot
Cb = (6.00, 2.05)
ex, ey, ez = (0.78, 0.13), (-1.62, 0.09), (0.0, -1.55)     # forward, left, up
Lx, Ly, Lz, ZG = 1.35, 0.55, 0.21, -1.45
ROLL = math.radians(-12.0)                                 # left (near) side sinks
KBEND = -0.45                                              # knee offset ALONG body x

def P(a_, b_, c_, roll=ROLL):
    yb = b_ * math.cos(roll) - c_ * math.sin(roll)
    zb = b_ * math.sin(roll) + c_ * math.cos(roll)
    return (Cb[0] + a_ * ex[0] + yb * ey[0] + zb * ez[0],
            Cb[1] + a_ * ex[1] + yb * ey[1] + zb * ez[1])

FLT, FRT, HLT, HRT = P(Lx, Ly, Lz), P(Lx, -Ly, Lz), P(-Lx, Ly, Lz), P(-Lx, -Ly, Lz)
FLB, FRB, HLB, HRB = P(Lx, Ly, -Lz), P(Lx, -Ly, -Lz), P(-Lx, Ly, -Lz), P(-Lx, -Ly, -Lz)
FEET = {'FL': P(Lx, Ly, ZG, 0.0), 'FR': P(Lx, -Ly, ZG, 0.0),
        'HL': P(-Lx, Ly, ZG, 0.0), 'HR': P(-Lx, -Ly, ZG, 0.0)}
HIPS = {'FL': FLB, 'FR': FRB, 'HL': HLB, 'HR': HRB}
legs = {}
for nm in FEET:
    hip, foot = HIPS[nm], FEET[nm]
    v = (foot[0] - hip[0], foot[1] - hip[1])
    legs[nm] = (hip, (hip[0] + 0.5 * v[0] + KBEND * ex[0],
                      hip[1] + 0.5 * v[1] + KBEND * ex[1]), foot)

feet = [FEET[n] for n in ('HL', 'FL', 'FR', 'HR')]
gcx = sum(q[0] for q in feet) / 4.0; gcy = sum(q[1] for q in feet) / 4.0
poly([(gcx + 1.22 * (q[0] - gcx), gcy + 1.22 * (q[1] - gcy)) for q in feet], 'gGround')

for nm in ('HR', 'HL'):                                    # far legs and their joints first
    hip, knee, foot = legs[nm]
    poly([hip, knee, foot], 'gLimbBack', closed=False)
    circle(knee[0], knee[1], 0.068, 'gJointBack')
    circle(hip[0], hip[1], 0.068, 'gJointBack')
for nm in ('FR', 'FL'):
    hip, knee, foot = legs[nm]
    poly([hip, knee, foot], 'gLimb', closed=False)

poly([HLT, FLT, FRT, HRT], 'gBodyTop')
poly([HLT, FLT, FLB, HLB], 'gBodySide')
poly([FLT, FRT, FRB, FLB], 'gBodyFront')
poly([P(-Lx, Ly, Lz, 0.0), P(Lx, Ly, Lz, 0.0),
      P(Lx, -Ly, Lz, 0.0), P(-Lx, -Ly, Lz, 0.0)], 'gLevel')   # level roof

for nm in ('FR', 'FL'):                                    # near joints sit on top
    hip, knee, foot = legs[nm]
    circle(knee[0], knee[1], 0.068, 'gJoint')
    circle(hip[0], hip[1], 0.068, 'gJoint')

# ---------------------------------------------------------------- attitude
exh, eyh, ezh = norm(ex), norm(ey), (0.0, -1.0)
arc(4.10, 0.62, 0.28, exh, ezh, -35, 215, 'gArc')          # pitch: about lateral axis
line(4.10 - 0.62 * eyh[0], 0.62 - 0.62 * eyh[1],
     4.10 + 0.62 * eyh[0], 0.62 + 0.62 * eyh[1], 'gAxis')
label(3.41, 0.52, [(u'\u03c8', 'var'), ('p', 'sub')], w=0.7)

arc(7.80, 0.62, 0.28, eyh, ezh, -35, 215, 'gArc')          # roll: about forward axis
line(7.80 - 0.62 * exh[0], 0.62 - 0.62 * exh[1],
     7.80 + 0.62 * exh[0], 0.62 + 0.62 * exh[1], 'gAxis')
label(8.52, 0.52, [(u'\u03c8', 'var'), ('r', 'sub')], w=0.7)

# ---------------------------------------------------------------- joint angles -> leg
line(2.80, 4.42, 5.82, 4.52, 'gArrow')
label(5.38, 4.22, [(u'\u03c6', 'var'), ('i', 'sub')], w=0.66)

# ---------------------------------------------------------------- corrections
DELTA = {'FL': (1, +0.30, +0.46, (+0.44, +0.10)),
         'HL': (3, -0.30, +0.46, (-0.50, -0.04)),
         'FR': (2, +0.30, -0.20, (+0.46, +0.08)),
         'HR': (4, -0.30, -0.20, (-0.48, -0.22))}
for nm, (idx, dx, amp, loff) in DELTA.items():
    hip, knee, foot = legs[nm]
    mx = (knee[0] + foot[0]) / 2.0 + dx
    my = (knee[1] + foot[1]) / 2.0
    line(mx, my + amp / 2.0, mx, my - amp / 2.0,
         'gDeltaBig' if amp > 0 else 'gDeltaSmall')
    label(mx + loff[0], my + loff[1], [(u'\u0394', 'var'), (str(idx), 'sub')], w=0.62)

fc = ((FLT[0] + FRT[0] + FRB[0] + FLB[0]) / 4.0, (FLT[1] + FRT[1] + FRB[1] + FLB[1]) / 4.0)
line(fc[0] + 0.85 * exh[0], fc[1] + 0.85 * exh[1],
     fc[0] + 1.55 * exh[0], fc[1] + 1.55 * exh[1], 'gArrow')
label(8.46, 2.12, [(u'p\u0307', 'var'), ('x', 'sub')], w=0.7)

for nm in ('FL', 'FR', 'HL', 'HR'):
    hip, knee, foot = legs[nm]
    print('  %s hip (%.2f,%.2f) knee (%.2f,%.2f) foot (%.2f,%.2f) len %.2f'
          % (nm, hip[0], hip[1], knee[0], knee[1], foot[0], foot[1],
             math.hypot(foot[0] - hip[0], foot[1] - hip[1])))
print('trunk top y %.2f  fc (%.2f,%.2f)' % (min(q[1] for q in (FLT, FRT, HLT, HRT)), fc[0], fc[1]))

# ---------------------------------------------------------------- document
NS = ('xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
      'xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" '
      'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
      'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" '
      'xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0" '
      'xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0" '
      'xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0" '
      'xmlns:xlink="http://www.w3.org/1999/xlink"')
ST = 'svg:stroke-linejoin="round" svg:stroke-linecap="round"'
gstyles = {
 'gSwing': 'draw:fill="solid" draw:fill-color="#e7eef5" draw:stroke="none"',
 'gGround': 'draw:fill="solid" draw:fill-color="#eceff2" draw:stroke="none"',
 'gLevel': 'draw:fill="none" draw:stroke="dash" draw:stroke-dash="Dsh2" svg:stroke-width="0.026cm" svg:stroke-color="#8c99a6"',
 'gBodyTop': 'draw:fill="solid" draw:fill-color="#dce5ee" draw:stroke="solid" svg:stroke-width="0.022cm" svg:stroke-color="#26323f" ' + ST,
 'gBodySide': 'draw:fill="solid" draw:fill-color="#c2cfdd" draw:stroke="solid" svg:stroke-width="0.022cm" svg:stroke-color="#26323f" ' + ST,
 'gBodyFront': 'draw:fill="solid" draw:fill-color="#a9bacb" draw:stroke="solid" svg:stroke-width="0.022cm" svg:stroke-color="#26323f" ' + ST,
 'gLimb': 'draw:fill="none" draw:stroke="solid" svg:stroke-width="0.055cm" svg:stroke-color="#26323f" ' + ST,
 'gLimbBack': 'draw:fill="none" draw:stroke="solid" svg:stroke-width="0.048cm" svg:stroke-color="#98a5b2" ' + ST,
 'gJointBack': 'draw:fill="solid" draw:fill-color="#ffffff" draw:stroke="solid" svg:stroke-width="0.020cm" svg:stroke-color="#8d9aa8"',
 'gJoint': 'draw:fill="solid" draw:fill-color="#ffffff" draw:stroke="solid" svg:stroke-width="0.022cm" svg:stroke-color="#26323f"',
 'gHead': 'draw:fill="solid" draw:fill-color="#12496f" draw:stroke="none"',
 'gArrow': 'draw:fill="none" draw:stroke="solid" svg:stroke-width="0.026cm" svg:stroke-color="#1f5c8b" draw:marker-end="Ah" draw:marker-end-width="0.17cm" draw:marker-end-center="false" ' + ST,
 'gArc': 'draw:fill="none" draw:stroke="solid" svg:stroke-width="0.026cm" svg:stroke-color="#1f5c8b" draw:marker-end="Ah" draw:marker-end-width="0.16cm" draw:marker-end-center="false" ' + ST,
 'gAxisArr': 'draw:fill="none" draw:stroke="solid" svg:stroke-width="0.020cm" svg:stroke-color="#5a6875" draw:marker-end="Ah" draw:marker-end-width="0.14cm" draw:marker-end-center="false" ' + ST,
 'gDeltaBig': 'draw:fill="none" draw:stroke="solid" svg:stroke-width="0.030cm" svg:stroke-color="#b0562a" draw:marker-end="Ah" draw:marker-end-width="0.19cm" draw:marker-end-center="false" ' + ST,
 'gDeltaSmall': 'draw:fill="none" draw:stroke="solid" svg:stroke-width="0.024cm" svg:stroke-color="#b0562a" draw:marker-end="Ah" draw:marker-end-width="0.15cm" draw:marker-end-center="false" ' + ST,
 'gAxis': 'draw:fill="none" draw:stroke="dash" draw:stroke-dash="Dsh" svg:stroke-width="0.018cm" svg:stroke-color="#94a1ae" ' + ST,
}
gstyles.update(extra_styles)
auto = ['<style:style style:name="%s" style:family="graphic">'
        '<style:graphic-properties %s/></style:style>' % (n, p) for n, p in gstyles.items()]
for nm, fill in (('gText', 'draw:fill="none"'), ('gTextBg', 'draw:fill="solid" draw:fill-color="#ffffff"')):
    auto.append('<style:style style:name="%s" style:family="graphic">'
                '<style:graphic-properties %s draw:stroke="none" fo:padding-top="0cm" '
                'fo:padding-bottom="0cm" fo:padding-left="0cm" fo:padding-right="0cm" '
                'draw:auto-grow-width="false" draw:auto-grow-height="false" '
                'draw:textarea-horizontal-align="center" draw:textarea-vertical-align="middle"/>'
                '</style:style>' % (nm, fill))
auto.append('<style:style style:name="Plbl" style:family="paragraph">'
            '<style:paragraph-properties fo:text-align="center" fo:margin-top="0cm" '
            'fo:margin-bottom="0cm"/><style:text-properties fo:font-family="Liberation Serif" '
            'fo:font-size="8pt"/></style:style>')
for nm, (sz, it, pos) in {'Tvar': ('8pt', 'italic', None), 'Trm': ('8pt', 'normal', None),
                          'Tsub': ('6pt', 'italic', '-33% 58%'), 'Tsup': ('6pt', 'italic', '33% 58%'),
                          'Tsubrm': ('6pt', 'normal', '-33% 58%'),
                          'Tsuprm': ('6pt', 'normal', '33% 58%')}.items():
    p_ = ('fo:font-family="Liberation Serif" fo:font-size="%s" fo:font-style="%s" '
          'fo:color="#16202b"' % (sz, it)) + (' style:text-position="%s"' % pos if pos else '')
    auto.append('<style:style style:name="%s" style:family="text">'
                '<style:text-properties %s/></style:style>' % (nm, p_))
auto.append('<style:style style:name="PM1dp" style:family="drawing-page">'
            '<style:drawing-page-properties draw:fill="solid" draw:fill-color="#ffffff" '
            'presentation:background-visible="true" presentation:background-objects-visible="true"/>'
            '</style:style>')

doc = ('<?xml version="1.0" encoding="UTF-8"?>\n'
       '<office:document %s office:version="1.3" '
       'office:mimetype="application/vnd.oasis.opendocument.presentation">\n'
       '<office:styles>\n'
       '<draw:marker draw:name="Ah" svg:viewBox="0 0 20 24" svg:d="M10 0 L0 24 L20 24 Z"/>\n'
       '<draw:stroke-dash draw:name="Dsh" draw:style="round" draw:dots1="1" '
       'draw:dots1-length="0.075cm" draw:distance="0.075cm"/>\n'
       '<draw:stroke-dash draw:name="Dsh2" draw:style="round" draw:dots1="1" '
       'draw:dots1-length="0.12cm" draw:distance="0.09cm"/>\n'
       '</office:styles>\n<office:automatic-styles>\n'
       '<style:page-layout style:name="PM1"><style:page-layout-properties '
       'fo:page-width="%.2fcm" fo:page-height="%.2fcm" style:print-orientation="landscape" '
       'fo:margin-top="0cm" fo:margin-bottom="0cm" fo:margin-left="0cm" fo:margin-right="0cm"/>'
       '</style:page-layout>\n%s\n</office:automatic-styles>\n<office:master-styles>\n'
       '<style:master-page style:name="Default" style:page-layout-name="PM1" '
       'draw:style-name="PM1dp"/>\n</office:master-styles>\n'
       '<office:body><office:presentation>\n'
       '<draw:page draw:name="schematic" draw:style-name="PM1dp" draw:master-page-name="Default">\n'
       '%s\n</draw:page>\n</office:presentation></office:body>\n</office:document>\n'
       % (NS, W, H, '\n'.join(auto), '\n'.join(shapes)))

io.open(sys.argv[1], 'w', encoding='utf-8').write(doc)
print('wrote', sys.argv[1], len(shapes), 'shapes')
