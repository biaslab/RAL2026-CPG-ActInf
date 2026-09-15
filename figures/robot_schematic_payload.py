# -*- coding: utf-8 -*-
"""Add a strapped-down payload box, sliding laterally, to the edited schematic."""
import math, io, re, sys

src, dst = sys.argv[1], sys.argv[2]
s = io.open(src, encoding='utf-8').read()

# --- projection recovered from the existing trunk polygons
Cb = (6.00, 2.05)
ex, ey, ez = (0.78, 0.13), (-1.62, 0.09), (0.0, -1.55)
Lx, Ly, Lz = 1.35, 0.55, 0.21
ROLL = math.radians(-12.0)
cR, sR = math.cos(ROLL), math.sin(ROLL)

def P(a, b, c):
    yb = b * cR - c * sR
    zb = b * sR + c * cR
    return (Cb[0] + a * ex[0] + yb * ey[0] + zb * ez[0],
            Cb[1] + a * ex[1] + yb * ey[1] + zb * ez[1])

# --- payload: sits on the trunk roof, slid toward the (sinking) left side
BX, BY = 0.00, 0.36                 # centre offset: slightly rear, well off-centre
PXH, PYH, PZH = 0.62, 0.30, 0.20     # half extents
ZC = Lz + PZH                        # rests on the roof

def Q(sa, sb, sc):
    return P(BX + sa * PXH, BY + sb * PYH, ZC + sc * PZH)

TFL, TFR, THL, THR = Q(1, 1, 1), Q(1, -1, 1), Q(-1, 1, 1), Q(-1, -1, 1)
BFL, BFR, BHL, BHR = Q(1, 1, -1), Q(1, -1, -1), Q(-1, 1, -1), Q(-1, -1, -1)

# --- straps: over the payload, down the near face, wrapping under the trunk
def strap(ax):
    return [P(ax, Ly - 0.22, -Lz - 0.05), P(ax, Ly + 0.03, -Lz - 0.03),
            P(ax, Ly + 0.03, Lz + 0.01),
            P(ax, BY + PYH, Lz + 0.01), P(ax, BY + PYH, ZC + PZH + 0.015),
            P(ax, BY - PYH - 0.02, ZC + PZH + 0.015)]
STRAPS = [strap(BX - 0.36), strap(BX + 0.34)]

# --- slide arrow: along +y (lateral), lying just above the payload roof
AR0 = P(BX, BY - PYH + 0.22, ZC + PZH + 0.32)
AR1 = P(BX, BY + PYH + 0.60, ZC + PZH + 0.32)

allpts = [TFL, TFR, THL, THR, BFL, BFR, BHL, BHR, AR0, AR1] + sum(STRAPS, [])
print('payload+annot extent: x %.2f..%.2f  y %.2f..%.2f'
      % (min(p[0] for p in allpts), max(p[0] for p in allpts),
         min(p[1] for p in allpts), max(p[1] for p in allpts)))
print('roof corners  FL%s FR%s HL%s HR%s' % (TFL, TFR, THL, THR))
print('left overhang beyond trunk: %.2f cm' % ((BY + PYH - Ly) * math.hypot(*ey)))

# ---------------------------------------------------------------- emit XML
def poly(pts, style, closed=True):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, y0 = min(xs), min(ys)
    w = max(max(xs) - x0, 0.001); h = max(max(ys) - y0, 0.001)
    d = ' '.join('%d,%d' % (round((p[0] - x0) * 1000), round((p[1] - y0) * 1000)) for p in pts)
    t = 'polygon' if closed else 'polyline'
    return ('<draw:%s draw:style-name="%s" draw:layer="layout" svg:x="%.3fcm" svg:y="%.3fcm" '
            'svg:width="%.3fcm" svg:height="%.3fcm" svg:viewBox="0 0 %d %d" draw:points="%s">'
            '<text:p/></draw:%s>' % (t, style, x0, y0, w, h,
                                     int(round(w * 1000)), int(round(h * 1000)), d, t))

def line(p0, p1, style):
    return ('<draw:line draw:style-name="%s" draw:layer="layout" svg:x1="%.3fcm" svg:y1="%.3fcm" '
            'svg:x2="%.3fcm" svg:y2="%.3fcm"><text:p/></draw:line>'
            % (style, p0[0], p0[1], p1[0], p1[1]))

PAD = ('fo:padding-top="0.136cm" fo:padding-bottom="0.136cm" fo:padding-left="0.263cm" '
       'fo:padding-right="0.263cm"')
NEW_STYLES = {
    'grPayTop':   'draw:stroke="solid" svg:stroke-width="0.022cm" svg:stroke-color="#3b3228" '
                  'svg:stroke-linecap="round" draw:fill="solid" draw:fill-color="#e8e0d3" ' + PAD,
    'grPaySide':  'draw:stroke="solid" svg:stroke-width="0.022cm" svg:stroke-color="#3b3228" '
                  'svg:stroke-linecap="round" draw:fill="solid" draw:fill-color="#d6cbb8" ' + PAD,
    'grPayFront': 'draw:stroke="solid" svg:stroke-width="0.022cm" svg:stroke-color="#3b3228" '
                  'svg:stroke-linecap="round" draw:fill="solid" draw:fill-color="#c3b69f" ' + PAD,
    'grStrap':    'draw:stroke="solid" svg:stroke-width="0.085cm" svg:stroke-color="#3f4650" '
                  'svg:stroke-linejoin="round" svg:stroke-linecap="butt" draw:fill="none" ' + PAD,
    'grSlide':    'draw:stroke="solid" svg:stroke-width="0.040cm" svg:stroke-color="#9c3b2e" '
                  'draw:marker-end="Ah" draw:marker-end-width="0.24cm" '
                  'draw:marker-end-center="false" svg:stroke-linecap="round" draw:fill="none" ' + PAD,
}
styles = ''.join('<style:style style:name="%s" style:family="graphic" '
                 'style:parent-style-name="standard"><style:graphic-properties %s/>'
                 '</style:style>' % (n, p) for n, p in NEW_STYLES.items())

shapes = (poly([THL, TFL, TFR, THR], 'grPayTop')
          + poly([THL, TFL, BFL, BHL], 'grPaySide')
          + poly([TFL, TFR, BFR, BFL], 'grPayFront')
          + ''.join(poly(p, 'grStrap', closed=False) for p in STRAPS)
          + line(AR0, AR1, 'grSlide'))

s = s.replace('</office:automatic-styles>', styles + '</office:automatic-styles>', 1)

# insert before the level-roof reference so that dashed outline stays on top
m = re.search(r'<draw:polygon draw:style-name="gr10"[^>]*>.*?</draw:polygon>', s, re.S)
assert m, 'level-roof polygon not found'
s = s[:m.end()] + shapes + s[m.end():]

io.open(dst, 'w', encoding='utf-8').write(s)
print('wrote', dst)
