# -*- coding: utf-8 -*-
"""two_layer.fodp -- agent / controller / robot block diagram, editable in Impress.

Usage: python3 two_layer_gen.py ../figures/two_layer.fodp
Page is 8.6 cm wide so it drops into a single IEEE column at ~1:1.
"""
import io, sys

W, H = 8.30, 6.55
shapes = []

# ---------------------------------------------------------------- primitives
def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

def rect(x0, y0, x1, y1, style, r=0.10):
    shapes.append('<draw:rect draw:style-name="%s" draw:corner-radius="%.2fcm" '
                  'svg:x="%.3fcm" svg:y="%.3fcm" svg:width="%.3fcm" svg:height="%.3fcm">'
                  '<text:p/></draw:rect>' % (style, r, x0, y0, x1 - x0, y1 - y0))

def line(x1, y1, x2, y2, style):
    shapes.append('<draw:line draw:style-name="%s" svg:x1="%.3fcm" svg:y1="%.3fcm" '
                  'svg:x2="%.3fcm" svg:y2="%.3fcm"><text:p/></draw:line>'
                  % (style, x1, y1, x2, y2))

def polyline(pts, style):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, y0 = min(xs), min(ys)
    w = max(max(xs) - x0, 0.001); h = max(max(ys) - y0, 0.001)
    d = ' '.join('%d,%d' % (round((p[0] - x0) * 1000), round((p[1] - y0) * 1000)) for p in pts)
    shapes.append('<draw:polyline draw:style-name="%s" svg:x="%.3fcm" svg:y="%.3fcm" '
                  'svg:width="%.3fcm" svg:height="%.3fcm" svg:viewBox="0 0 %d %d" '
                  'draw:points="%s"><text:p/></draw:polyline>'
                  % (style, x0, y0, w, h, int(round(w * 1000)), int(round(h * 1000)), d))

def circle(cx, cy, r, style):
    shapes.append('<draw:ellipse draw:style-name="%s" svg:x="%.3fcm" svg:y="%.3fcm" '
                  'svg:width="%.3fcm" svg:height="%.3fcm"><text:p/></draw:ellipse>'
                  % (style, cx - r, cy - r, 2 * r, 2 * r))

KIND = {'var': 'Tvar', 'rm': 'Trm', 'sub': 'Tsub', 'sup': 'Tsup',
        'subrm': 'Tsubrm', 'suprm': 'Tsuprm', 'grey': 'Tgrey', 'greyvar': 'Tgreyvar'}

def textbox(cx, cy, lines, w, h, gst='gText'):
    """lines: list of (runs, paragraph-style); runs: list of (text, kind)."""
    body = ''.join('<text:p text:style-name="%s">%s</text:p>'
                   % (ps, ''.join('<text:span text:style-name="%s">%s</text:span>'
                                  % (KIND[k], esc(t)) for t, k in runs))
                   for runs, ps in lines)
    shapes.append('<draw:frame draw:style-name="%s" svg:x="%.3fcm" svg:y="%.3fcm" '
                  'svg:width="%.3fcm" svg:height="%.3fcm">'
                  '<draw:text-box>%s</draw:text-box></draw:frame>'
                  % (gst, cx - w / 2, cy - h / 2, w, h, body))

def label(cx, cy, runs, w=0.55, h=0.32, ps='Plbl'):
    textbox(cx, cy, [(runs, ps)], w, h)

def labelbg(cx, cy, runs, w=0.55, h=0.32, ps='Plbl'):
    textbox(cx, cy, [(runs, ps)], w, h, gst='gTextBg')

# ---------------------------------------------------------------- geometry
CX = 4.95                      # centre of the block column
BW = 4.20                      # block width
BH = 0.78                      # block height
BX0, BX1 = CX - BW / 2, CX + BW / 2
GX0, GX1 = BX0 - 0.25, BX1 + 0.25
LANE = 1.35                    # x of the feedback lane
LBX = LANE + 0.34              # x of the labels on the feedback lane

AY0, AY1 = 0.64, 0.64 + BH     # agent block
BY0, BY1 = 2.44, 2.44 + BH     # cpg block
CY0, CY1 = 3.90, 3.90 + BH     # servo block
DY0, DY1 = 5.50, 5.50 + BH     # robot block

GA = (0.22, 1.55)              # agent layer panel
GB = (2.02, 4.90)              # controller layer panel

def blk(y0, y1, title, sub=None, style='gBlock'):
    rect(BX0, y0, BX1, y1, style)
    cy = (y0 + y1) / 2.0
    lines = [(title, 'Ptitle')]
    if sub:
        lines.append((sub, 'Psub'))
    textbox(CX, cy if sub is None else cy, lines, BW - 0.10, y1 - y0 - 0.06)

def down(y0, y1, runs):
    line(CX, y0, CX, y1, 'gArrow')
    label(CX + 0.30, (y0 + y1) / 2.0, runs, w=0.5)

# ---------------------------------------------------------------- panels
rect(GX0, GA[0], GX1, GA[1], 'gPanel', r=0.14)
rect(GX0, GB[0], GX1, GB[1], 'gPanel', r=0.14)
label(GX0 + 0.62, GA[0] + 0.22, [('agent layer', 'grey')], w=1.2, ps='Pgrp')
label(GX0 + 0.78, GB[0] + 0.22, [('controller layer', 'grey')], w=1.55, ps='Pgrp')

# ---------------------------------------------------------------- blocks
blk(AY0, AY1, [('Active inference', 'rm')],
    [('minimize expected free energy', 'grey')])
blk(BY0, BY1, [('Central pattern generator', 'rm')],
    [('oscillator states ', 'grey'), (u'z', 'greyvar')])
blk(CY0, CY1, [('Joint servo controllers', 'rm')],
    [('PD tracking of ', 'grey'), (u'φ', 'greyvar')])
blk(DY0, DY1, [('Robot', 'rm')],
    [('base state ', 'grey'), (u'ξ', 'greyvar')], style='gPlant')

# ---------------------------------------------------------------- forward path
down(AY1, BY0, [(u'θ', 'var')])
down(BY1, CY0, [(u'φ', 'var')])
down(CY1, DY0, [(u'τ', 'var')])

# ---------------------------------------------------------------- feedback path
NY = 3.55                                        # noise injection node
polyline([(BX0, (DY0 + DY1) / 2.0), (LANE, (DY0 + DY1) / 2.0),
          (LANE, (AY0 + AY1) / 2.0), (BX0, (AY0 + AY1) / 2.0)], 'gArrow')
circle(LANE, NY, 0.14, 'gNode')
label(LANE, NY - 0.005, [('+', 'rm')], w=0.30, h=0.28)
line(0.42, NY, LANE - 0.20, NY, 'gArrowThin')
label(0.78, NY - 0.29, [('noise', 'grey')], w=0.90, ps='Pgrp')
labelbg(LBX, 4.55, [(u'ξ', 'var')])
labelbg(LBX, 2.55, [('x', 'var')])

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
 'gPanel': 'draw:fill="solid" draw:fill-color="#f1f5f9" draw:stroke="dash" '
           'draw:stroke-dash="Dsh2" svg:stroke-width="0.020cm" svg:stroke-color="#a7b3bf"',
 'gBlock': 'draw:fill="solid" draw:fill-color="#ffffff" draw:stroke="solid" '
           'svg:stroke-width="0.026cm" svg:stroke-color="#26323f" ' + ST,
 'gPlant': 'draw:fill="solid" draw:fill-color="#dce5ee" draw:stroke="solid" '
           'svg:stroke-width="0.026cm" svg:stroke-color="#26323f" ' + ST,
 'gNode': 'draw:fill="solid" draw:fill-color="#ffffff" draw:stroke="solid" '
          'svg:stroke-width="0.024cm" svg:stroke-color="#1f5c8b"',
 'gArrow': 'draw:fill="none" draw:stroke="solid" svg:stroke-width="0.028cm" '
           'svg:stroke-color="#1f5c8b" draw:marker-end="Ah" draw:marker-end-width="0.18cm" '
           'draw:marker-end-center="false" ' + ST,
 'gArrowThin': 'draw:fill="none" draw:stroke="solid" svg:stroke-width="0.022cm" '
               'svg:stroke-color="#8c99a6" draw:marker-end="Ah" draw:marker-end-width="0.14cm" '
               'draw:marker-end-center="false" ' + ST,
}
auto = ['<style:style style:name="%s" style:family="graphic">'
        '<style:graphic-properties %s/></style:style>' % (n, p) for n, p in gstyles.items()]
for nm, fill in (('gText', 'draw:fill="none"'),
                 ('gTextBg', 'draw:fill="solid" draw:fill-color="#ffffff"')):
    auto.append('<style:style style:name="%s" style:family="graphic">'
                '<style:graphic-properties %s draw:stroke="none" fo:padding-top="0cm" '
                'fo:padding-bottom="0cm" fo:padding-left="0cm" fo:padding-right="0cm" '
                'draw:auto-grow-width="false" draw:auto-grow-height="false" '
                'draw:textarea-horizontal-align="center" draw:textarea-vertical-align="middle"/>'
                '</style:style>' % (nm, fill))
for nm, sz in (('Plbl', '8pt'), ('Ptitle', '8pt'), ('Psub', '7pt'), ('Pgrp', '7pt')):
    auto.append('<style:style style:name="%s" style:family="paragraph">'
                '<style:paragraph-properties fo:text-align="center" fo:margin-top="0cm" '
                'fo:margin-bottom="0cm"/><style:text-properties '
                'fo:font-family="Liberation Serif" fo:font-size="%s"/></style:style>' % (nm, sz))
TXT = {'Tvar': ('8pt', 'italic', '#16202b', None), 'Trm': ('8pt', 'normal', '#16202b', None),
       'Tsub': ('6pt', 'italic', '#16202b', '-33% 58%'),
       'Tsup': ('6pt', 'italic', '#16202b', '33% 58%'),
       'Tsubrm': ('6pt', 'normal', '#16202b', '-33% 58%'),
       'Tsuprm': ('6pt', 'normal', '#16202b', '33% 58%'),
       'Tgrey': ('7pt', 'normal', '#6b7885', None),
       'Tgreyvar': ('7pt', 'italic', '#6b7885', None)}
for nm, (sz, it, col, pos) in TXT.items():
    p_ = ('fo:font-family="Liberation Serif" fo:font-size="%s" fo:font-style="%s" '
          'fo:color="%s"' % (sz, it, col)) + (' style:text-position="%s"' % pos if pos else '')
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
       '<draw:stroke-dash draw:name="Dsh2" draw:style="round" draw:dots1="1" '
       'draw:dots1-length="0.10cm" draw:distance="0.09cm"/>\n'
       '</office:styles>\n<office:automatic-styles>\n'
       '<style:page-layout style:name="PM1"><style:page-layout-properties '
       'fo:page-width="%.2fcm" fo:page-height="%.2fcm" style:print-orientation="landscape" '
       'fo:margin-top="0cm" fo:margin-bottom="0cm" fo:margin-left="0cm" fo:margin-right="0cm"/>'
       '</style:page-layout>\n%s\n</office:automatic-styles>\n<office:master-styles>\n'
       '<style:master-page style:name="Default" style:page-layout-name="PM1" '
       'draw:style-name="PM1dp"/>\n</office:master-styles>\n'
       '<office:body><office:presentation>\n'
       '<draw:page draw:name="twolayer" draw:style-name="PM1dp" draw:master-page-name="Default">\n'
       '%s\n</draw:page>\n</office:presentation></office:body>\n</office:document>\n'
       % (NS, W, H, '\n'.join(auto), '\n'.join(shapes)))

io.open(sys.argv[1], 'w', encoding='utf-8').write(doc)
print('wrote', sys.argv[1], len(shapes), 'shapes')
