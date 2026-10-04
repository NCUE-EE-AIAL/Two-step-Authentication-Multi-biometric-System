"""Regenerate editable, white-background research SVGs with Python 3 + pycairo.

Run from any directory: python3 doc/generate_figures.py
Geometry is deliberately fixed for stable paper/README layouts. Cairo measures
Arial text before export; overflowing labels fail instead of being compressed.
Sources: doc/Flow_diagram.png, doc/graph_hr.png, train_face.ipynb,
         src/models.py, src/constants.py, src/triplet_loss.py.
"""
from pathlib import Path
from xml.sax.saxutils import escape

import cairo

OUT = Path(__file__).resolve().parent
INK = '#202020'
MUTED = '#505050'


class Figure:
    def __init__(self, name, height, title, description):
        self.name, self.height = name, height
        self.ctx = cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1))
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" '
            f'viewBox="0 0 1200 {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(description)}</desc>',
            '<defs><marker id="arrow" viewBox="0 0 10 8" refX="10" refY="4" '
            'markerWidth="8" markerHeight="6.4" orient="auto-start-reverse" '
            'markerUnits="userSpaceOnUse"><path d="M0 0L10 4L0 8Z" fill="#202020"/></marker></defs>',
            f'<rect id="background" width="1200" height="{height}" fill="#ffffff"/>',
        ]

    def text(self, x, y, text, size=17, bold=False, anchor='start', width=None, muted=False):
        self.ctx.select_font_face('Arial', cairo.FONT_SLANT_NORMAL,
                                  cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
        self.ctx.set_font_size(size)
        ext = self.ctx.text_extents(text)
        if width is not None:
            assert ext.width <= width, (self.name, text, ext.width, width)
        left = x - (ext.x_advance / 2 if anchor == 'middle' else ext.x_advance if anchor == 'end' else 0)
        assert left >= 0 and left + ext.width <= 1200, (text, left, ext.width)
        assert y - size >= 0 and y + 5 <= self.height, (text, y)
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
                          f'font-size="{size}" font-weight="{700 if bold else 400}" '
                          f'text-anchor="{anchor}" fill="{MUTED if muted else INK}">{escape(text)}</text>')

    def path(self, d, arrow=True, dashed=False, light=False):
        self.parts.append(f'<path d="{d}" fill="none" stroke="{"#b0b0b0" if light else INK}" '
                          f'stroke-width="{1 if light else 1.5}" stroke-linejoin="round" '
                          + ('stroke-dasharray="6 5" ' if dashed else '')
                          + ('marker-end="url(#arrow)" ' if arrow else '') + '/>')

    def node(self, x, y, w, h, title, details=(), shape='rect'):
        style = f'fill="#ffffff" stroke="{INK}" stroke-width="1.5"'
        if shape == 'diamond':
            self.parts.append(f'<polygon points="{x+w/2},{y} {x+w},{y+h/2} {x+w/2},{y+h} {x},{y+h/2}" {style}/>')
            available = w * .65
        elif shape == 'input':
            self.parts.append(f'<polygon points="{x+18},{y} {x+w},{y} {x+w-18},{y+h} {x},{y+h}" {style}/>')
            available = w - 46
        else:
            radius = h / 2 if shape == 'terminal' else 0
            self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" {style}/>')
            available = w - 20
        lines = [(title, 18, True)] + [(line, 16, False) for line in details]
        baseline = y + h / 2 - (len(lines)-1)*11 + 6
        for i, (line, size, bold) in enumerate(lines):
            self.text(x+w/2, baseline+i*22, line, size, bold, 'middle', available)

    def heading(self, y, letter, title, subtitle=None):
        self.text(36, y, f'({letter})', 20, True)
        self.text(76, y, title, 20, True)
        if subtitle:
            self.text(76, y+27, subtitle, 16, muted=True)

    def rule(self, y):
        self.path(f'M36 {y}H1164', arrow=False, light=True)

    def save(self):
        (OUT/self.name).write_text('\n'.join(self.parts + ['</svg>']) + '\n')
        print(f'Wrote {self.name} (1200 × {self.height})')


def workflow():
    f = Figure('workflow.svg', 570, 'Two-step biometric authentication workflow',
               'Redrawn from the original project workflow. A face match selects the enrolled voice reference. '
               'The recorded voice and selected reference enter speaker verification. Either failed decision denies access. '
               'White background and black lines remain fixed in all display themes. The integrated controller is a proposed design.')
    f.heading(48, 'a', 'Face identification', 'Identify a candidate among the enrolled users.')
    f.node(36, 112, 122, 64, 'Login', shape='terminal')
    f.node(190, 112, 180, 64, 'Image capture', shape='input')
    f.node(408, 99, 254, 90, 'Face identification', ['MTCNN → VGG16', 'Candidate identity i*'])
    f.node(728, 91, 240, 106, 'Face match?', ['max p(i | x) > 0.75'], 'diamond')
    f.node(1042, 112, 122, 64, 'Denied', shape='terminal')
    for d in ['M158 144H190', 'M370 144H408', 'M662 144H728', 'M968 144H1042']:
        f.path(d)
    f.text(1004, 132, 'No', 16, anchor='middle')
    # Preserve the original workflow's identity-dependent reference selection.
    f.path('M848 197V228H535V254')
    f.text(864, 222, 'Yes', 16)
    f.node(408, 254, 254, 68, 'Select voice reference', ['Enrolled embedding for i*'])
    f.heading(344, 'b', 'Speaker verification')
    f.node(54, 389, 282, 78, 'Audio recording', ['Claimed user’s speech'], 'input')
    f.node(408, 378, 254, 100, 'Speaker verification', ['ResCNN embeddings', 'Cosine similarity s'])
    f.node(728, 375, 240, 106, 'Voice match?', ['s > threshold τ'], 'diamond')
    f.node(1042, 396, 122, 64, 'Pass', shape='terminal')
    for d in ['M336 428H408', 'M535 322V378', 'M662 428H728', 'M968 428H1042', 'M848 375V296H1103V176']:
        f.path(d)
    f.text(1003, 415, 'Yes', 16, anchor='middle')
    f.text(869, 284, 'No', 16)
    f.rule(512)
    f.text(36, 541, 'System design: both gates must pass. The released code evaluates the two branches separately.', 16)
    f.save()


def overview():
    f = Figure('architecture.svg', 590, 'Face and voice model architecture summary',
               'Two aligned model summaries. Face images are cropped to 224 by 224, processed by VGG16 and a dense classifier. '
               'Speech is represented as 160 by 64 filter-bank features and processed by four ResCNN stages to a 512-dimensional embedding. '
               'Dimensions omit the batch axis. All backgrounds and nodes are white.')
    f.heading(46, 'a', 'Face identification', 'VGGFace-initialized VGG16 with a five-class classification head')
    xs = [36, 228, 420, 612, 804, 996]
    face = [
        ('Image', ['H × W × 3']),
        ('MTCNN crop', ['Resize', '224 × 224 × 3']),
        ('VGG16 base', ['13 convolutions', '7 × 7 × 512']),
        ('Flatten', ['25 088']),
        ('Dense head', ['256 → 128', 'ReLU activations']),
        ('Dense + softmax', ['5 probabilities', 'Threshold > 0.75']),
    ]
    for i,(title,lines) in enumerate(face):
        f.node(xs[i], 112, 168, 104, title, lines)
        if i: f.path(f'M{xs[i]-24} 164H{xs[i]}')
    f.text(36, 248, 'Head: Dense 256 → Dropout (0.5) → Dense 128 → Dense 5; L2 regularization on hidden dense layers.', 16)
    f.text(36, 274, 'Training: freeze the convolutional base for 10 epochs, then fine-tune all layers for 10 epochs (Adam, LR 10⁻⁵).', 16)
    f.rule(301)
    f.heading(340, 'b', 'Speaker verification', 'Residual CNN with cosine triplet loss; independent of the face classifier')
    voice = [
        ('Speech', ['16 kHz mono', 'Voice activity filter']),
        ('Fbank features', ['64 filters', '160 × 64 × 1']),
        ('ResCNN × 4', ['64 / 128 / 256 / 512', '10 × 4 × 512']),
        ('Temporal mean', ['Reshape: 10 × 2048', '2048']),
        ('Dense + L2 norm', ['512-dimensional', 'Unit-length vector']),
        ('Cosine score', ['Reference + query', 'Threshold decision']),
    ]
    for i,(title,lines) in enumerate(voice):
        f.node(xs[i], 405, 168, 104, title, lines)
        if i: f.path(f'M{xs[i]-24} 457H{xs[i]}')
    f.text(36, 546, 'Shapes omit the batch axis. Voice scoring uses recorded pairs in the released evaluation code.', 16)
    f.text(36, 573, 'Architecture sources: train_face.ipynb, src/models.py and src/constants.py.', 16, muted=True)
    f.save()


def voice_model():
    f = Figure('voice_architecture.svg', 1040, 'ResCNN speaker encoder: architecture, residual blocks and training objective',
               'Panel a traces the 160 by 64 input through stages with 64, 128, 256 and 512 channels, temporal averaging, '
               'a 512-unit dense layer and L2 normalization. Panel b expands one downsampling stage. Panel c expands the '
               'three-convolution identity block with its skip connection. Panel d states cosine triplet loss for 32 triplets. '
               'Tensor dimensions omit batch size; white background is fixed.')
    f.heading(44, 'a', 'Speaker encoder', 'Input: normalized 64-filter Fbank features, 160 frames (approximately 1.6 s)')
    xs = [36, 268, 500, 732, 964]
    first = [('Input', ['160 × 64 × 1']),
             ('Stage 1', ['64 channels', '80 × 32 × 64']),
             ('Stage 2', ['128 channels', '40 × 16 × 128']),
             ('Stage 3', ['256 channels', '20 × 8 × 256']),
             ('Stage 4', ['512 channels', '10 × 4 × 512'])]
    for i,(title,lines) in enumerate(first):
        f.node(xs[i], 104, 200, 96, title, lines)
        if i: f.path(f'M{xs[i]-32} 152H{xs[i]}')
    f.path('M1064 200V228H136V256')
    second = [('Reshape', ['10 × 2048']), ('Mean over time', ['2048']),
              ('Dense', ['512 units']), ('L2 normalization', ['z = h / ‖h‖₂']), ('Embedding', ['512 dimensions', '‖z‖₂ = 1'])]
    for i,(title,lines) in enumerate(second):
        f.node(xs[i], 256, 200, 88, title, lines)
        if i: f.path(f'M{xs[i]-32} 300H{xs[i]}')
    f.text(36, 374, 'Tensor order: time × frequency × channels. Batch dimension omitted; every stage halves time and frequency.', 16)
    f.rule(397)
    f.heading(436, 'b', 'Residual stage', 'Same structure at all four stages; C = 64, 128, 256 or 512 output channels')
    stage = [('Conv 5 × 5', ['C filters; stride 2', 'Same padding']),
             ('BatchNorm', ['Channel normalization']),
             ('Clipped ReLU', ['min(max(x, 0), 20)']),
             ('Identity block × 3', ['C channels', 'Expanded in panel (c)']),
             ('Stage output', ['T/2 × F/2 × C'])]
    for i,(title,lines) in enumerate(stage):
        f.node(xs[i], 490, 200, 94, title, lines)
        if i: f.path(f'M{xs[i]-32} 537H{xs[i]}')
    f.rule(609)
    f.heading(648, 'c', 'Identity block', 'Spatial dimensions and channel count are preserved; all convolutions use stride 1.')
    f.text(518, 710, 'Identity shortcut', 16, anchor='middle')
    f.path('M94 784V725H884V766')
    f.parts.append('<circle cx="94" cy="784" r="3" fill="#202020"/>')
    f.text(45, 790, 'x', 18)
    f.path('M66 784H140')
    f.node(140, 740, 200, 88, 'Conv 1 × 1', ['C filters', 'BN + clipped ReLU'])
    f.node(380, 740, 200, 88, 'Conv 3 × 3', ['C filters', 'BN + clipped ReLU'])
    f.node(620, 740, 200, 88, 'Conv 1 × 1', ['C filters', 'BatchNorm'])
    f.path('M340 784H380'); f.path('M580 784H620'); f.path('M820 784H866')
    f.parts.append(f'<circle cx="884" cy="784" r="18" fill="#ffffff" stroke="{INK}" stroke-width="1.5"/>')
    f.text(884, 791, '+', 24, anchor='middle')
    f.path('M902 784H944')
    f.node(944, 740, 220, 88, 'Clipped ReLU', ['Output: T × F × C'])
    f.rule(856)
    f.heading(895, 'd', 'Training objective')
    f.text(36, 928, 'Anchor a, same-speaker positive p, and different-speaker negative n share the encoder weights.', 17)
    f.text(600, 964, 'L = Σᵢ max[cos(aᵢ, nᵢ) − cos(aᵢ, pᵢ) + 0.1, 0]', 22, anchor='middle')
    f.text(36, 997, '32 triplets per batch (96 clips). Random triplets: epochs 0–20; hard-mined triplets: epochs 21–60.', 16)
    f.text(36, 1023, 'Sources: src/models.py, src/triplet_loss.py, src/constants.py and train_voice.py. BN = batch normalization.', 16, muted=True)
    f.save()


if __name__ == '__main__':
    workflow()
    overview()
    voice_model()
