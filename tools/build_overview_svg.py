import math
import random
import sys
from itertools import pairwise
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "docs" / "overview.svg"

D = 25.2  # loop duration (s)
W, H = 800, 392
X0, X1 = 64.0, 756.0
EASE = "0.42 0 0.58 1"
LIN = "0 0 1 1"
FADE = 0.18
MONO = 'font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"'

# logo colours as mid-tones: navy lightened, green as is, orange deepened
BLUE, BLUE_DK = "#2f7cba", "#4a92cf"
GREEN = "#1ba67d"
ORANGE, ORANGE_DK = "#e8861a", "#fd9a2b"
RED = "#d1495b"
FILL = {"b": BLUE, "g": GREEN, "o": "#d9770f"}  # theme-independent, white text >= 3:1

rng = random.Random(20261118)

STAGES = [
    (
        "Calls",
        "Variant calls",
        "MANTA and SMOOVE report call A, only DELLY reports call B",
    ),
    (
        "Evidence",
        "Alignment evidence",
        "split reads, discordant pairs, soft clips and a coverage drop at call A",
    ),
    (
        "Features",
        "Feature vector",
        "caller support, call attributes, alignment inside the call vs its flanks",
    ),
    (
        "Compare",
        "Model comparison",
        "six models tuned in nested CV vs the callers, McNemar / DeLong with Holm",
    ),
    (
        "Calibrate",
        "Calibration",
        "isotonic or sigmoid scaling brings the reliability curve onto the diagonal",
    ),
    (
        "Transfer",
        "Cross-dataset",
        "every train / test pair and leave-one-out, next to the feature shift",
    ),
    (
        "Verdict",
        "Verdict",
        "calibrated probability against the Youden threshold, explained by SHAP",
    ),
]
SW = [2.8, 6.9, 9.7, 15.1, 18.1, 21.1, 24.9]  # stage k ends at SW[k]

ACT1 = [(0, 1), (9.5, 1), (9.8, 0), (24.8, 0), (25.1, 1)]
ACT2 = [(0, 0), (9.75, 0), (10.05, 1), (24.55, 1), (24.85, 0)]
R1 = 10.5  # act 1 is hidden: reset its elements
R2 = 24.95  # act 2 is hidden: reset its elements


def f2(v):
    """Format a number with at most two decimals."""
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _val(v):
    """Format a keyframe value."""
    if isinstance(v, (tuple, list)):
        return " ".join(f2(x) for x in v)
    if isinstance(v, (int, float)):
        return f2(v)
    return v


def _kt(t):
    """Convert a time in seconds to a SMIL keyTimes value."""
    if t <= 0:
        return "0"
    if t >= D:
        return "1"
    return f"{t / D:.4f}".rstrip("0").rstrip(".")


def anim(attr, kfs, tag="animate", extra="", spline=EASE):
    """Return a looping SMIL animation from absolute (time, value) keyframes."""
    kfs = list(kfs)
    if kfs[0][0] > 0:
        kfs.insert(0, (0.0, kfs[0][1]))
    if kfs[-1][0] < D:
        kfs.append((D, kfs[-1][1]))
    for (ta, _), (tb, _) in pairwise(kfs):
        assert tb > ta, (attr, kfs)
    vals = [_val(v) for _, v in kfs]
    splines = [LIN if a == b else spline for a, b in pairwise(vals)]
    extra = f" {extra}" if extra else ""
    return (
        f'<{tag} attributeName="{attr}"{extra} dur="{f2(D)}s" '
        'repeatCount="indefinite" calcMode="spline" '
        f'keyTimes="{";".join(_kt(t) for t, _ in kfs)}" '
        f'keySplines="{";".join(splines)}" values="{";".join(vals)}"/>'
    )


def op(kfs):
    """Return an opacity animation."""
    return anim("opacity", kfs)


def move(kfs, spline=EASE):
    """Return a translate animation."""
    return anim("transform", kfs, "animateTransform", 'type="translate"', spline)


def scale(kfs):
    """Return a scale animation."""
    return anim("transform", kfs, "animateTransform", 'type="scale"')


def appear(t, reset, dur=0.2, level=1.0):
    """Return opacity keyframes: hidden, fade in at t, hidden again at reset."""
    return [(0, 0), (t, 0), (t + dur, level), (reset, level), (reset + 0.05, 0)]


def window(t_in, t_out):
    """Return opacity keyframes of an element shown between t_in and t_out."""
    return [(0, 0), (t_in, 0), (t_in + 0.25, 1), (t_out, 1), (t_out + 0.2, 0)]


def held(v0, v1, a, b, reset):
    """Return keyframes going from v0 to v1 between a and b, back at reset."""
    return [(0, v0), (a, v0), (b, v1), (reset, v1), (reset + 0.05, v0)]


def grow(attr, a, b, v0, v1, reset):
    """Return an animation of attr from v0 to v1 between a and b."""
    return anim(attr, held(v0, v1, a, b, reset))


def pop(t, reset):
    """Return a scale animation that pops in at t with a small overshoot."""
    return scale(
        [
            (0, 0.001),
            (t, 0.001),
            (t + 0.2, 1.2),
            (t + 0.32, 1),
            (reset, 1),
            (reset + 0.05, 0.001),
        ]
    )


def animated(el, *anims):
    """Insert SMIL animations into an element."""
    a = "".join(anims)
    if el.endswith("/>"):
        tag = el[1 : el.index(" ")]
        return f"{el[:-2]}>{a}</{tag}>"
    close = el.rindex("</")
    return el[:close] + a + el[close:]


def stage_opacity(k):
    """Return opacity keyframes of the stage-k text (crossfade at SW)."""
    if k == 0:
        return [(0, 1), (SW[0] - FADE, 1), (SW[0], 0), (SW[-1], 0), (SW[-1] + FADE, 1)]
    a, b = SW[k - 1], SW[k]
    return [(0, 0), (a, 0), (a + FADE, 1), (b - FADE, 1), (b, 0)]


def text(x, y, s, cls="fg", size=10, extra=""):
    """Return a text element."""
    extra = f" {extra}" if extra else ""
    return f'<text class="{cls}" x="{f2(x)}" y="{f2(y)}" font-size="{f2(size)}"{extra}>{s}</text>'


def rect(x, y, w, h, extra=""):
    """Return a rect element."""
    extra = f" {extra}" if extra else ""
    return f'<rect x="{f2(x)}" y="{f2(y)}" width="{f2(w)}" height="{f2(h)}"{extra}/>'


def title(x, y, s):
    """Return a panel title."""
    return text(x, y, s, "mu", 11, 'font-weight="600" letter-spacing="1"')


def pill(cx, cy, s, kfs, fill="b"):
    """Return a solid label pill with white text, shown by kfs."""
    w = 14 + 6.1 * len(s)
    return (
        '<g opacity="0">'
        + op(kfs)
        + rect(cx - w / 2, cy - 7.5, w, 15, f'rx="7.5" fill="{FILL[fill]}"')
        + text(cx, cy + 3.7, s, "wt", 10.5, 'font-weight="600" text-anchor="middle"')
        + "</g>"
    )


def popped(cx, cy, t, reset, body):
    """Return body drawn around (cx, cy) and popping in at t."""
    return (
        f'<g transform="translate({f2(cx)},{f2(cy)})"><g transform="scale(0.001)">'
        + pop(t, reset)
        + body
        + "</g></g>"
    )


# act 1: genome ruler, caller tracks, read pileup, coverage, features
TX0, TX1 = 140.0, 756.0
PILE_X1 = 540.0  # right edge of the pileup
BP_L, BP_R = 250.0, 430.0
CALL_A = (BP_L + BP_R) / 2
CALL_B = (612.0, 668.0)
RULER_Y, LABEL_Y = 61.0, 81.0
CALLER_Y = {"MANTA": 96.0, "DELLY": 116.0, "SMOOVE": 136.0}
CALLS = {"MANTA": [(246.0, 433.0)], "DELLY": [CALL_B], "SMOOVE": [(254.0, 426.0)]}
MISSED = {"MANTA": [CALL_B], "DELLY": [(BP_L, BP_R)], "SMOOVE": [CALL_B]}
RL, RH = 30.0, 4.4
PY0, PITCH, NROWS = 150.0, 7.6, 12
SPLIT_ROWS, DISC_ROWS = (0, 1, 2), (5, 6, 7)
COV_BASE, COV_H = 270.0, 26.0
FY, FH = 294.0, 18.0

T_SCAN = (0.25, 2.0)
T_BADGE = 2.2
T_REVEAL = (2.9, 3.5)
T_SPLIT, T_DISC, T_CLIP, T_COV = 3.65, 4.3, 4.95, 5.75
T_DIM = 6.95
T_FLY, FLY_DUR, FLY_GAP = 7.1, 0.6, 0.1
T_GROUPS = 9.05


def scan_t(x):
    """Return the time at which the scan line reaches x."""
    a, b = T_SCAN
    return a + (x - TX0) / (TX1 - TX0) * (b - a)


def row_y(r):
    """Return the top of pileup row r."""
    return PY0 + r * PITCH


def _fits(row, a, b):
    """Tell whether the span a..b fits in a pileup row."""
    return all(b + 3 <= c or d + 3 <= a for c, d in row)


def make_pileup():
    """Return the packed reads as (row, kind, segments, link) tuples.

    A segment is (x0, x1, style) with style "n" (aligned), "low" (low MAPQ) or
    "clip" (soft-clipped). A link is (x0, x1) and joins split pieces or mates.
    """
    special = []
    for _ in range(2):
        s = BP_L - rng.uniform(9, 22)
        rest = s + RL - BP_L
        special.append(("split", [(s, BP_L, "n"), (BP_R, BP_R + rest, "n")]))
    s = BP_R - rng.uniform(9, 22)
    left = BP_R - s
    special.append(("split", [(BP_L - left, BP_L, "n"), (BP_R, s + RL, "n")]))
    for _ in range(3):
        a = BP_L - RL - rng.uniform(6, 70)
        b = BP_R + rng.uniform(6, 70)
        special.append(("disc", [(a, a + RL, "n"), (b, b + RL, "n")]))

    rows = [[] for _ in range(NROWS)]
    reads = []
    for r, (kind, segs) in zip(SPLIT_ROWS + DISC_ROWS, special, strict=True):
        rows[r].append((segs[0][0], segs[-1][1]))
        reads.append((r, kind, segs, (segs[0][1], segs[1][0])))

    starts = sorted(rng.uniform(TX0 - RL / 2, PILE_X1 - RL / 2) for _ in range(300))
    for s in starts:
        e = s + RL
        if s < BP_L < e:
            kind, segs = "clip", [(s, BP_L, "n"), (BP_L, e, "clip")]
        elif s < BP_R < e:
            kind, segs = "clip", [(s, BP_R, "clip"), (BP_R, e, "n")]
        elif BP_L <= s and e <= BP_R:
            if rng.random() > 0.12:
                continue
            kind, segs = "low", [(s, e, "low")]
        else:
            kind, segs = "n", [(s, e, "n")]
        for r in range(NROWS):
            if _fits(rows[r], s, e):
                rows[r].append((s, e))
                reads.append((r, kind, segs, None))
                break
    return reads


def coverage(reads):
    """Return bin centres and smoothed depth of the aligned (not clipped) reads."""
    xs = [TX0 + 4 * i for i in range(int((PILE_X1 - TX0) / 4) + 1)]
    depth = [
        sum(
            1
            for _, _, segs, _ in reads
            for a, b, style in segs
            if style != "clip" and a <= x < b
        )
        for x in xs
    ]
    smooth = []
    for i in range(len(depth)):
        win = depth[max(0, i - 1) : i + 2]
        smooth.append(sum(win) / len(win))
    return xs, smooth


def read_rects(reads, kinds, styles, cls, extra=""):
    """Return the rects of the given read kinds and segment styles."""
    out = []
    for r, kind, segs, _ in reads:
        if kind not in kinds:
            continue
        for a, b, style in segs:
            if style in styles:
                op_ = ' fill-opacity=".3"' if style == "low" else ""
                out.append(
                    rect(a, row_y(r), b - a, RH, f'rx="1" class="{cls}"{op_}{extra}')
                )
    return out


def read_links(reads, kind, cls):
    """Return the lines joining split pieces or discordant mates."""
    dash = ' stroke-dasharray="3 2"' if kind == "disc" else ""
    out = []
    for r, k, _, link in reads:
        if k == kind and link is not None:
            y = row_y(r) + RH / 2
            out.append(
                f'<path class="{cls}" d="M{f2(link[0])},{f2(y)}H{f2(link[1])}" '
                f'stroke-width="1"{dash}/>'
            )
    return out


def highlight(reads, kind, styles, t):
    """Return the highlighted reads of one evidence kind, with a short halo."""
    o = ['<g opacity="0">' + op(appear(t, R1))]
    if kind in ("split", "disc"):
        o += read_links(reads, kind, "bs")
    o += read_rects(reads, {kind}, styles, "b")
    o.append("</g>")
    halo = [(0, 0), (t, 0), (t + 0.15, 0.9), (t + 0.65, 0), (R1, 0)]
    o.append('<g opacity="0" fill="none" stroke-width="2.2">' + op(halo))
    for r, k, segs, _ in reads:
        if k != kind:
            continue
        for a, b, style in segs:
            if style in styles:
                o.append(
                    rect(a - 1, row_y(r) - 1, b - a + 2, RH + 2, 'rx="2" class="bs"')
                )
    o.append("</g>")
    return o


LENS_X, LENS_Y, LENS_W, LENS_H = 556.0, 150.0, 200.0, 86.0
BASE_X0, BASE_DX, N_BASES, N_DEL = 592.0, 7.9, 20, 9
LENS_READS = [0, 2, 4, 1]  # first base of each read in the lens


def build_lens(reads):
    """Return the zoom on the right breakpoint, with the soft-clipped bases."""
    clips = [
        (r, segs) for r, k, segs, _ in reads if k == "clip" and segs[0][2] == "clip"
    ]
    r_a = min(clips, key=lambda c: abs(c[0] - 5))[0]
    ay = row_y(r_a) + RH / 2
    ref = [rng.choice("ACGT") for _ in range(N_BASES)]
    other = [rng.choice([b for b in "ACGT" if b != ref[i]]) for i in range(N_DEL)]

    def bases(y, seq):
        spans = []
        for i, (base, clipped, dim) in seq:
            x = BASE_X0 + i * BASE_DX
            cls = f"n{base}" if clipped else "mu"
            fo = ' fill-opacity=".45"' if dim else ""
            spans.append(f'<tspan x="{f2(x)}" class="{cls}"{fo}>{base}</tspan>')
        return (
            f'<text y="{f2(y)}" font-size="10" font-weight="600" {MONO} '
            f'text-anchor="middle">{"".join(spans)}</text>'
        )

    show = window(T_CLIP, T_DIM)
    o = ['<g opacity="0">' + op(show)]
    o.append(
        f'<path class="bs" d="M{f2(BP_R + 4)},{f2(ay)}L{f2(LENS_X)},{f2(LENS_Y + LENS_H / 2)}" '
        'stroke-width="1" stroke-dasharray="3 2" fill="none"/>'
    )
    o.append(
        f'<circle class="bs" cx="{f2(BP_R)}" cy="{f2(ay)}" r="5" fill="none" stroke-width="1.5"/>'
    )
    o.append(
        rect(
            LENS_X,
            LENS_Y,
            LENS_W,
            LENS_H,
            'rx="8" class="b bs" fill-opacity=".05" stroke-width="1.2"',
        )
    )
    o.append(
        '<defs><clipPath id="vt-type">'
        + animated(
            rect(LENS_X, LENS_Y, 0, LENS_H),
            grow("width", T_CLIP + 0.3, T_CLIP + 0.85, 0, LENS_W, R1),
        )
        + "</clipPath></defs>"
    )
    o.append('<g clip-path="url(#vt-type)">')
    o.append(text(LENS_X + 10, LENS_Y + 25, "ref", "mu", 10))
    o.append(
        bases(LENS_Y + 25, [(i, (b, False, i < N_DEL)) for i, b in enumerate(ref)])
    )
    for k, first in enumerate(LENS_READS):
        seq = []
        for i in range(first, N_BASES):
            if i < N_DEL:
                seq.append((i, (other[i], True, False)))
            else:
                seq.append((i, (ref[i], False, False)))
        o.append(bases(LENS_Y + 40 + 12 * k, seq))
    o.append("</g>")
    bx = BASE_X0 + (N_DEL - 0.5) * BASE_DX
    o.append(
        f'<path class="bs" d="M{f2(bx)},{f2(LENS_Y + 14)}V{f2(LENS_Y + LENS_H - 6)}" '
        'stroke-width="1" stroke-dasharray="2 2"/>'
    )
    o.append("</g>")
    cx = BASE_X0 + (N_DEL - 1) / 2 * BASE_DX
    o.append(pill(cx, LENS_Y, "soft clips", show))
    return o


FEATS = [
    ("MANTA", "1", 1.0),
    ("DELLY", "0", 0.0),
    ("SMOOVE", "1", 1.0),
    ("MAXQV", "32", 0.85),
    ("SVLEN", "-18k", 0.9),
    ("CG", "0.47", 0.22),
    ("coverage", None, 0.12),
    ("insert", "2.1k", 0.9),
    ("MAPQ", "14", 0.25),
    ("clipped", None, 0.85),
    ("split", "3", 1.0),
    ("discord.", "3", 0.92),
    ("Δinsert", "+1.6k", 0.88),
    ("ΔMAPQ", "-42", 0.85),
]  # cell shade: >= .8 dark with white text, <= .3 light, both >= 3:1
FEAT_GROUPS = [(0, 3, "caller support"), (3, 6, "call"), (6, 14, "inside vs flanks")]


def feature_sources(reads):
    """Return the point of the evidence each feature is computed from."""
    first = {}
    for r, kind, segs, _ in reads:
        first.setdefault(kind, (r, segs))
    split_y = row_y(SPLIT_ROWS[1]) + RH / 2
    disc_y = row_y(DISC_ROWS[1]) + RH / 2
    low_r, low_segs = first["low"]
    clip_l = next(
        (r, segs) for r, k, segs, _ in reads if k == "clip" and segs[1][2] == "clip"
    )
    return [
        (CALL_A, CALLER_Y["MANTA"]),
        (CALL_A, CALLER_Y["DELLY"]),
        (CALL_A, CALLER_Y["SMOOVE"]),
        (275.0, CALLER_Y["MANTA"]),
        (CALL_A, LABEL_Y - 4),
        (CALL_A + 40, RULER_Y),
        (CALL_A, COV_BASE - 3),
        (CALL_A, row_y(DISC_ROWS[0]) + RH / 2),
        ((low_segs[0][0] + low_segs[0][1]) / 2, row_y(low_r) + RH / 2),
        (BP_L + 5, row_y(clip_l[0]) + RH / 2),
        (CALL_A - 30, split_y),
        (CALL_A + 30, disc_y),
        (BP_L - 30, row_y(9)),
        (BP_R + 30, row_y(10)),
    ]


def build_act1():
    """Return the SVG lines of act 1 (calls, evidence, features)."""
    reads = make_pileup()
    xs, depth = coverage(reads)
    o = ['<g opacity="1">' + op(ACT1)]
    add = o.append

    add(
        text(
            TX0 - 12,
            RULER_Y + 4,
            "chr1",
            "fg",
            11,
            'text-anchor="end" font-weight="600"',
        )
    )
    ticks = "".join(
        f"M{f2(x)},{f2(RULER_Y)}v{3 if (x - TX0) % 100 else 6}"
        for x in (TX0 + 20 * i for i in range(int((TX1 - TX0) / 20) + 1))
    )
    add(
        f'<path class="ax" d="M{f2(TX0)},{f2(RULER_Y)}H{f2(TX1)}{ticks}" '
        'stroke-width="1" fill="none" opacity=".7"/>'
    )
    for k in range(6):
        s = f"{12.40 + 0.01 * k:.2f}" + (" Mb" if k == 0 else "")
        add(text(TX0 + 100 * k + 3, RULER_Y - 4, s, "mu", 10))

    for name, y in CALLER_Y.items():
        add(text(TX0 - 12, y + 4, name, "fg", 11, 'text-anchor="end"'))
        add(
            f'<path class="ax" d="M{f2(TX0)},{f2(y)}H{f2(TX1)}" stroke-width="1" '
            'opacity=".45"/>'
        )

    add(
        '<g opacity="0">'
        + op(
            [
                (0, 0),
                (T_SCAN[0] - 0.1, 0),
                (T_SCAN[0], 1),
                (T_SCAN[1], 1),
                (T_SCAN[1] + 0.3, 0),
            ]
        )
        + "<g>"
        + move(held((0, 0), (TX1 - TX0, 0), T_SCAN[0], T_SCAN[1], R1), LIN)
        + rect(
            TX0 - 26,
            LABEL_Y + 2,
            26,
            CALLER_Y["SMOOVE"] - LABEL_Y + 6,
            'fill="url(#vt-trail)"',
        )
        + f'<path class="bs" d="M{f2(TX0)},{f2(LABEL_Y + 2)}V{f2(CALLER_Y["SMOOVE"] + 8)}" '
        'stroke-width="1.6"/>' + "</g></g>"
    )
    for name, calls in CALLS.items():
        y = CALLER_Y[name]
        for a, b in calls:
            t = scan_t(a)
            add(
                '<g opacity="0">'
                + op(appear(t, R1, dur=0.25))
                + "<g>"
                + move(held((0, -9), (0, 0), t, t + 0.3, R1))
                + rect(a, y - 6, b - a, 12, 'rx="2.5" class="b"')
                + text(
                    (a + b) / 2,
                    y + 3.6,
                    "DEL",
                    "wt",
                    10,
                    'font-weight="700" text-anchor="middle"',
                )
                + "</g></g>"
            )
    for name, spans in MISSED.items():
        y = CALLER_Y[name]
        for a, b in spans:
            add(
                animated(
                    rect(
                        a,
                        y - 6,
                        b - a,
                        12,
                        'rx="2.5" class="ax" fill="none" stroke-width="1" '
                        'stroke-dasharray="3 3" opacity="0"',
                    ),
                    op(appear(scan_t(a), R1, level=0.75)),
                )
            )
    for label, cx, x_in, support in (
        ("call A", CALL_A, 246.0, "2/3 callers"),
        ("call B", sum(CALL_B) / 2, CALL_B[0], "1/3 callers"),
    ):
        add(
            animated(
                text(
                    cx,
                    LABEL_Y,
                    label,
                    "fg",
                    11,
                    'font-weight="600" text-anchor="middle" opacity="0"',
                ),
                op(appear(scan_t(x_in), R1)),
            )
        )
        w = 14 + 5.9 * len(support)
        add(
            popped(
                cx + 24 + w / 2,
                LABEL_Y - 4,
                T_BADGE + (0.15 if label == "call B" else 0),
                R1,
                rect(-w / 2, -7.5, w, 15, 'class="pill" rx="7.5"')
                + text(
                    0, 3.6, support, "b", 10, 'font-weight="600" text-anchor="middle"'
                ),
            )
        )

    add(
        '<defs><clipPath id="vt-reveal">'
        + animated(
            rect(TX0, PY0 - 4, 0, COV_BASE - PY0 + 6),
            grow("width", *T_REVEAL, 0, PILE_X1 - TX0, R1),
        )
        + "</clipPath></defs>"
    )
    dim = [(0, 1), (T_DIM, 1), (T_DIM + 0.3, 0.35), (R1, 0.35), (R1 + 0.05, 1)]
    for label, y0, y1 in (
        ("reads", PY0 - 3, row_y(NROWS) - 2),
        ("coverage", COV_BASE - COV_H - 1, COV_BASE),
    ):
        add(
            rect(
                TX0, y0, PILE_X1 - TX0, y1 - y0, 'rx="3" class="mu" fill-opacity=".06"'
            )
        )
        add(text(TX0 - 12, (y0 + y1) / 2 + 4, label, "fg", 11, 'text-anchor="end"'))
    add('<g clip-path="url(#vt-reveal)">' + op(dim))
    add('<g opacity=".6">')
    o += read_links(reads, "split", "ax")
    o += read_links(reads, "disc", "ax")
    o += read_rects(reads, {"n", "clip", "low", "split", "disc"}, {"n", "low"}, "mu")
    o += read_rects(reads, {"clip"}, {"clip"}, "mu")
    add("</g>")
    o += highlight(reads, "split", {"n"}, T_SPLIT)
    o += highlight(reads, "disc", {"n"}, T_DISC)
    o += highlight(reads, "clip", {"clip"}, T_CLIP)

    k = COV_H / max(depth)
    pts = "".join(
        f"L{f2(x)},{f2(COV_BASE - d * k)}" for x, d in zip(xs, depth, strict=True)
    )
    area = f"M{f2(TX0)},{f2(COV_BASE)}{pts}L{f2(PILE_X1)},{f2(COV_BASE)}Z"
    add(
        f'<g transform="translate(0,{f2(COV_BASE)})"><g transform="scale(1,0.001)">'
        + scale(held((1, 0.001), (1, 1), T_COV, T_COV + 0.45, R1))
        + f'<g transform="translate(0,{f2(-COV_BASE)})">'
        + f'<path class="b bs" d="{area}" fill-opacity=".28" stroke-width="1" stroke-linejoin="round"/>'
        + "</g></g></g>"
    )
    add("</g>")
    add(
        f'<path class="ax" d="M{f2(TX0)},{f2(COV_BASE)}H{f2(PILE_X1)}" stroke-width="1" '
        'opacity=".6"/>'
    )
    add(
        f'<path class="bs" d="M{f2(BP_L)},{f2(LABEL_Y + 6)}V{f2(COV_BASE)}M{f2(BP_R)},'
        f'{f2(LABEL_Y + 6)}V{f2(COV_BASE)}" stroke-width="1" stroke-dasharray="2 3" '
        'fill="none" opacity="0">' + op(appear(T_SPLIT, R1, level=0.8)) + "</path>"
    )
    add(
        pill(
            CALL_A,
            row_y(SPLIT_ROWS[-1] + 1.4) + 2,
            "split reads",
            window(T_SPLIT, T_DIM),
        )
    )
    add(
        pill(
            CALL_A,
            row_y(DISC_ROWS[-1] + 1.4) + 2,
            "discordant pairs",
            window(T_DISC, T_DIM),
        )
    )
    add(pill(CALL_A, COV_BASE - 10, "coverage drop", window(T_COV + 0.35, T_DIM)))
    o += build_lens(reads)

    add('<g opacity="0">' + op(appear(T_DIM + 0.05, R1)))
    for a, b, label in (
        (BP_L - 60, BP_L, "flank"),
        (BP_L, BP_R, "inside"),
        (BP_R, BP_R + 60, "flank"),
    ):
        add(
            f'<path class="bs" d="M{f2(a + 2)},{f2(COV_BASE + 3)}v3H{f2(b - 2)}v-3" fill="none" '
            'stroke-width="1.2"/>'
        )
        add(text((a + b) / 2, COV_BASE + 17, label, "fg", 10, 'text-anchor="middle"'))
    add("</g>")

    inside = [d for x, d in zip(xs, depth, strict=True) if BP_L + 8 <= x <= BP_R - 8]
    flanks = [
        d
        for x, d in zip(xs, depth, strict=True)
        if BP_L - 60 <= x < BP_L - 8 or BP_R + 8 < x <= BP_R + 60
    ]
    cov_scale = 30 / (sum(flanks) / len(flanks))
    n_clip = sum(1 for _, kind, _, _ in reads if kind == "clip")
    values = {
        "coverage": f"{sum(inside) / len(inside) * cov_scale:.0f}x",
        "clipped": str(n_clip),
    }
    cw = (TX1 - TX0) / len(FEATS)
    add(
        animated(
            text(
                TX0 - 12,
                FY + 13,
                "call A",
                "mu",
                11,
                'text-anchor="end" font-style="italic" opacity="0"',
            ),
            op(appear(T_FLY, R1)),
        )
    )
    for i, ((label, val, v), (sx, sy)) in enumerate(
        zip(FEATS, feature_sources(reads), strict=True)
    ):
        x = TX0 + i * cw
        t0 = T_FLY + FLY_GAP * i
        t1 = t0 + FLY_DUR
        cls = "mu" if v == 0 else "b"
        add(
            f'<circle class="{cls}" r="3.2" opacity="0">'
            + op([(0, 0), (t0, 0), (t0 + 0.08, 1), (t1 - 0.05, 1), (t1 + 0.05, 0)])
            + move([(0, (sx, sy)), (t0, (sx, sy)), (t1, (x + cw / 2, FY + FH / 2))])
            + "</circle>"
        )
        if i < 3 and v == 0:
            cell = rect(
                x + 2,
                FY + 0.5,
                cw - 4,
                FH - 1,
                'rx="2" class="bs" fill="none" stroke-width="1"',
            )
        else:
            cell = rect(
                x + 2,
                FY,
                cw - 4,
                FH,
                f'rx="2" class="b" fill-opacity="{f2(max(v, 0.12))}"',
            )
        val = val or values[label]
        vcls = "wt" if v >= 0.8 else "fg"
        add(
            '<g opacity="0">'
            + op(appear(t1, R1, dur=0.12))
            + cell
            + text(
                x + cw / 2,
                FY + 12.8,
                val,
                vcls,
                10,
                'font-weight="600" text-anchor="middle"',
            )
            + text(x + cw / 2, FY + FH + 13, label, "fg", 10, 'text-anchor="middle"')
            + "</g>"
        )
    add('<g opacity="0">' + op(appear(T_GROUPS, R1)))
    gy = FY + FH + 19
    for a, b, label in FEAT_GROUPS:
        xa, xb = TX0 + a * cw + 3, TX0 + b * cw - 3
        add(
            f'<path class="ax" d="M{f2(xa)},{f2(gy)}v3H{f2(xb)}v-3" fill="none" stroke-width="1"/>'
        )
        add(
            text(
                (xa + xb) / 2,
                gy + 15,
                label,
                "mu",
                10.5,
                'text-anchor="middle" font-style="italic"',
            )
        )
    add("</g>")
    add("</g>")
    return o


# act 2, left panel: model comparison, full in stage 4, compact from stage 5
MODELS = [
    ("CatBoost", 0.905, 0.016, "ref"),
    ("XGBoost", 0.896, 0.017, "ns"),
    ("Random Forest", 0.884, 0.018, "*"),
    ("MLP", 0.861, 0.021, "**"),
    ("Logistic Regression", 0.823, 0.02, "***"),
    ("KNN", 0.801, 0.024, "***"),
]
START_SLOT = [3, 0, 5, 2, 1, 4]  # row of each model before sorting
BASELINES = [
    ("MANTA", 0.742, 0.019, "***"),
    ("DELLY OR SMOOVE", 0.694, 0.02, "***"),
    ("all callers AND", 0.655, 0.018, "***"),
]
BEST = MODELS[0][0]
PX0, PX1 = 196.0, 400.0
ROW_Y = [106.0 + 19 * i for i in range(6)]
SEP_Y = ROW_Y[-1] + 16
ROW_Y += [SEP_Y + 16 + 19 * j for j in range(3)]
AXIS_Y = ROW_Y[-1] + 18
ST_X, ST_C = 414.0, 246.0  # star column, full and compact
CV_X = 238.0  # right edge of the compact values
BAND_R, BAND_RC = 446.0, 276.0
NFOLD = 10
T_OUTER, T_FUNNEL = 10.15, 10.4
T_INNER = [10.65 + 0.2 * j for j in range(5)]  # inner validation fold moves
T_REFIT, T_SCORE = 11.7, 12.0
FOLD_T = [T_SCORE + 0.35 + 0.1 * i for i in range(NFOLD)]  # outer fold scored
T_CI, T_SORT, T_STARS, T_BEST = 13.45, 13.75, 14.5, 14.7
T_COMPACT = 15.15
FULL = [(0, 1), (T_COMPACT, 1), (T_COMPACT + 0.3, 0), (R2, 0), (R2 + 0.05, 1)]
COMPACT = [(0, 0), (T_COMPACT + 0.15, 0), (T_COMPACT + 0.4, 1), (R2, 1), (R2 + 0.05, 0)]


def px(v):
    """Convert a balanced accuracy to a horizontal pixel position."""
    return PX0 + (v - 0.5) / 0.5 * (PX1 - PX0)


def diamond(x, y, r, extra):
    """Return a diamond marker."""
    return (
        f'<path d="M{f2(x)},{f2(y - r)}L{f2(x + r)},{f2(y)}L{f2(x)},{f2(y + r)}'
        f'L{f2(x - r)},{f2(y)}Z" {extra}/>'
    )


def separator(xa, xb, lx):
    """Return the dashed separator above the caller baselines."""
    half = 44.0
    return (
        f'<path class="ax" d="M{f2(xa)},{f2(SEP_Y)}H{f2(lx - half)}M{f2(lx + half)},'
        f'{f2(SEP_Y)}H{f2(xb)}" stroke-width="1" stroke-dasharray="3 3" opacity=".6"/>'
        + text(
            lx,
            SEP_Y + 3.5,
            "caller baselines",
            "mu",
            10,
            'text-anchor="middle" font-style="italic"',
        )
    )


def build_compare(scores):
    """Return the SVG lines of the model comparison (left panel of act 2)."""
    o = []
    add = o.append
    rows = MODELS + BASELINES
    slide = move(held((0, 0), (ST_C - ST_X, 0), T_COMPACT, T_COMPACT + 0.35, R2))
    add(title(X0, 70, "MODEL COMPARISON"))
    add("<g>" + slide + text(ST_X, 89, "vs best", "mu", 10) + "</g>")

    add(
        '<g opacity="0">'
        + op(appear(T_BEST, R2, dur=0.25))
        + animated(
            rect(
                X0 - 6,
                ROW_Y[0] - 9,
                BAND_R - X0 + 6,
                18,
                f'rx="4" fill="{GREEN}" fill-opacity=".12" '
                f'stroke="{GREEN}" stroke-opacity=".55"',
            ),
            grow(
                "width",
                T_COMPACT,
                T_COMPACT + 0.35,
                BAND_R - X0 + 6,
                BAND_RC - X0 + 6,
                R2,
            ),
        )
        + "</g>"
    )

    add("<g>" + op(FULL))
    add(separator(X0, ST_X + 22, (PX0 + PX1) / 2 - 50))
    add('<g fill="none" stroke-width="1">')
    add(f'<path class="ax" d="M{f2(PX0)},{f2(AXIS_Y)}H{f2(PX1)}"/>')
    for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        add(f'<path class="ax" d="M{f2(px(v))},{f2(AXIS_Y)}v4"/>')
    add("</g>")
    for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        add(text(px(v), AXIS_Y + 16, f2(v), "mu", 10, 'text-anchor="middle"'))
    add(
        text(
            (PX0 + PX1) / 2,
            AXIS_Y + 31,
            "balanced accuracy, mean and 95% BCa bootstrap CI",
            "mu",
            10,
            'text-anchor="middle"',
        )
    )
    for x in (px(v) for v in (0.6, 0.7, 0.8, 0.9)):
        add(
            f'<path class="ax" d="M{f2(x)},{f2(ROW_Y[0] - 12)}V{f2(AXIS_Y)}" stroke-width="1" opacity=".18"/>'
        )
    add("</g>")
    add(
        '<g opacity="0">'
        + op(COMPACT)
        + separator(X0, BAND_RC, (X0 + BAND_RC) / 2)
        + "</g>"
    )

    for i, ((name, mean, half, star), y) in enumerate(zip(rows, ROW_Y, strict=True)):
        model = i < len(MODELS)
        g = ["<g>"]
        if model and START_SLOT[i] != i:
            dy = ROW_Y[START_SLOT[i]] - y
            g.append(move(held((0, dy), (0, 0), T_SORT, T_SORT + 0.6, R2)))
        g.append(text(X0, y + 4, name, "fg", 11))
        g.append("<g>" + op(FULL))
        cls = "b" if model else "mu"
        for k, (v, jit) in enumerate(scores[i]):
            t = FOLD_T[k]
            g.append(
                f'<circle class="{cls}" cx="{f2(px(v))}" cy="{f2(y + jit)}" '
                'r="1.8" fill-opacity=".6" opacity="0">'
                + op(
                    [
                        (0, 0),
                        (t, 0),
                        (t + 0.08, 1),
                        (T_CI, 1),
                        (T_CI + 0.25, 0.3),
                        (R2, 0.3),
                        (R2 + 0.05, 0),
                    ]
                )
                + "</circle>"
            )
        lo, hi = px(mean - half), px(mean + half)
        g.append('<g opacity="0">' + op(appear(T_CI, R2, dur=0.25)))
        g.append(
            f'<path class="{"bs" if model else "ax"}" d="M{f2(lo)},{f2(y)}H{f2(hi)}M{f2(lo)},'
            f'{f2(y - 3)}v6M{f2(hi)},{f2(y - 3)}v6" fill="none" stroke-width="1.4"/>'
        )
        if model:
            g.append(f'<circle class="b" cx="{f2(px(mean))}" cy="{f2(y)}" r="3.6"/>')
        else:
            g.append(
                diamond(px(mean), y, 4, 'class="ax" fill="none" stroke-width="1.4"')
            )
        g.append("</g>")
        if name == BEST:
            g.append(
                f'<circle cx="{f2(px(mean))}" cy="{f2(y)}" r="3.8" fill="{GREEN}" opacity="0">'
                + op(appear(T_BEST, R2, dur=0.25))
                + "</circle>"
            )
        g.append("</g>")
        vcls = "g" if name == BEST else "fg"
        g.append(
            '<g opacity="0">'
            + op(COMPACT)
            + text(
                CV_X,
                y + 4,
                f"{mean:.3f}",
                vcls,
                11,
                'font-weight="600" text-anchor="end"',
            )
            + "</g>"
        )
        scls = "g" if star == "ref" else "fg"
        g.append(
            '<g opacity="0">'
            + op(appear(T_STARS, R2))
            + "<g>"
            + slide
            + text(ST_X, y + 4, star, scls, 10.5, 'font-weight="600"')
            + "</g></g>"
        )
        g.append("</g>")
        add("".join(g))
    return o


# act 2, right panel
RX = 476.0  # stage 4, next to the full comparison
RX2 = 306.0  # stages 5 to 7, next to the compact comparison
OX0, OY, OPITCH, OW, OH = 556.0, 102.0, 18.0, 16.0, 14.0
IX0, IY, IPITCH, IW, IH = 520.0, 152.0, 46.4, 44.4, 14.0
NINNER = 5


def fold_scores():
    """Return the (score, jitter) of every row on every outer fold."""
    return [
        [(mean + rng.gauss(0, 0.024), rng.uniform(-3, 3)) for _ in range(NFOLD)]
        for _, mean, _, _ in MODELS + BASELINES
    ]


def slot_y(i):
    """Return the row height of row i before the models are sorted."""
    return ROW_Y[START_SLOT[i]] if i < len(MODELS) else ROW_Y[i]


def build_nested_cv(scores):
    """Return the SVG lines of the nested cross-validation diagram.

    The first outer fold runs slowly: its train folds open into the inner
    folds, Optuna tunes while the validation fold moves, the model is refit
    and the held-out score flies to the comparison. The other nine run fast.
    """
    o = [
        '<g opacity="1">'
        + op(
            [
                (0, 1),
                (T_COMPACT - 0.15, 1),
                (T_COMPACT + 0.05, 0),
                (R2, 0),
                (R2 + 0.1, 1),
            ]
        )
    ]
    add = o.append
    add(title(RX, 70, "NESTED CROSS-VALIDATION"))
    add(text(RX, OY + 11, "outer", "mu", 10.5))
    add(text(RX, IY + 11, "inner", "mu", 10.5))
    for k in range(NFOLD):
        add(rect(OX0 + k * OPITCH, OY, OW, OH, 'rx="2" class="b" fill-opacity=".2"'))

    for k in range(NFOLD):
        start = T_OUTER if k == 0 else FOLD_T[k] - 0.03
        end = FOLD_T[k + 1] - 0.03 if k < NFOLD - 1 else R2
        kfs = [(0, 0), (start, 0), (start + 0.03, 1), (end, 1), (end + 0.03, 0)]
        if k == 0:
            kfs[2] = (start + 0.2, 1)
        add(
            animated(
                text(RX, OY + 27, f"fold {k + 1}/{NFOLD}", "mu", 10, 'opacity="0"'),
                op(kfs),
            )
        )

    t_last = FOLD_T[1] - 0.05
    funnel = (
        f"M{f2(OX0 + OPITCH)},{f2(OY + OH + 2)}H{f2(OX0 + (NFOLD - 1) * OPITCH + OW)}"
        f"L{f2(IX0 + (NINNER - 1) * IPITCH + IW)},{f2(IY - 3)}H{f2(IX0)}Z"
    )
    add(
        f'<path class="b bs" d="{funnel}" fill-opacity=".08" stroke-opacity=".5" '
        'stroke-width="1" stroke-dasharray="3 2" opacity="0">'
        + op(
            [
                (0, 0),
                (T_FUNNEL, 0),
                (T_FUNNEL + 0.3, 1),
                (T_REFIT, 1),
                (T_REFIT + 0.25, 0.45),
                (t_last, 0.45),
                (t_last + 0.25, 0),
            ]
        )
        + "</path>"
    )

    icx = IX0 + ((NINNER - 1) * IPITCH + IW) / 2
    tcx = OX0 + OPITCH + ((NFOLD - 2) * OPITCH + OW) / 2
    sx = ((NFOLD - 2) * OPITCH + OW) / ((NINNER - 1) * IPITCH + IW)
    hl = [(0, (0, 0))]
    for j in range(1, NINNER):
        hl += [
            (T_INNER[j] - 0.12, ((j - 1) * IPITCH, 0)),
            (T_INNER[j], (j * IPITCH, 0)),
        ]
    prev = NINNER - 1
    for k in range(1, NFOLD):
        pos = k % NINNER
        hl += [(FOLD_T[k] - 0.06, (prev * IPITCH, 0)), (FOLD_T[k], (pos * IPITCH, 0))]
        prev = pos
    hl += [(R2, (prev * IPITCH, 0)), (R2 + 0.05, (0, 0))]
    add(
        '<g opacity="0">'
        + op(
            [
                (0, 0),
                (T_FUNNEL, 0),
                (T_FUNNEL + 0.2, 1),
                (T_REFIT, 1),
                (T_REFIT + 0.25, 0.45),
                (R2, 0.45),
                (R2 + 0.05, 0),
            ]
        )
        + "<g>"
        + move(held((tcx - icx, OY - IY), (0, 0), T_FUNNEL, T_FUNNEL + 0.4, R2))
        + f'<g transform="translate({f2(icx)},{f2(IY)})"><g>'
        + scale(held((sx, 0.6), (1, 1), T_FUNNEL, T_FUNNEL + 0.4, R2))
        + f'<g transform="translate({f2(-icx)},{f2(-IY)})">'
        + "".join(
            rect(IX0 + j * IPITCH, IY, IW, IH, 'rx="2" class="b" fill-opacity=".2"')
            for j in range(NINNER)
        )
        + '<g opacity="0">'
        + op(appear(T_INNER[0], R2, dur=0.15))
        + "<g>"
        + move(hl)
        + rect(IX0, IY, IW, IH, 'rx="2" class="mu" fill-opacity=".6"')
        + "</g></g>"
        + "</g></g></g></g></g>"
    )

    kfs = [(0, (0, 0))]
    for k in range(1, NFOLD):
        kfs += [(FOLD_T[k] - 0.06, ((k - 1) * OPITCH, 0)), (FOLD_T[k], (k * OPITCH, 0))]
    kfs += [(R2, ((NFOLD - 1) * OPITCH, 0)), (R2 + 0.05, (0, 0))]
    add(
        '<g opacity="0">'
        + op(appear(T_OUTER, R2))
        + "<g>"
        + move(kfs)
        + rect(OX0, OY, OW, OH, 'rx="2" class="b"')
        + text(
            OX0 + OW / 2,
            OY - 5,
            "test",
            "b",
            10,
            'text-anchor="middle" font-weight="600"',
        )
        + "</g></g>"
    )
    add(
        rect(
            OX0 + OPITCH - 2,
            OY - 2,
            (NFOLD - 2) * OPITCH + OW + 4,
            OH + 4,
            'rx="3" class="bs" fill="none" stroke-width="1.6" opacity="0"',
        )[:-2]
        + ">"
        + op(
            [
                (0, 0),
                (T_REFIT, 0),
                (T_REFIT + 0.15, 1),
                (T_SCORE + 0.1, 1),
                (T_SCORE + 0.35, 0),
            ]
        )
        + "</rect>"
    )
    add(
        animated(
            rect(
                OX0 - 2,
                OY - 2,
                OW + 4,
                OH + 4,
                'rx="3" class="bs" fill="none" stroke-width="2" opacity="0"',
            ),
            op([(0, 0), (T_SCORE, 0), (T_SCORE + 0.1, 1), (T_SCORE + 0.5, 0)]),
        )
    )
    x0, y0 = OX0 + OW / 2, OY + OH / 2
    for i, rows in enumerate(scores):
        v, jit = rows[0]
        cls = "b" if i < len(MODELS) else "mu"
        add(
            f'<circle class="{cls}" r="2.4" opacity="0">'
            + op(
                [
                    (0, 0),
                    (T_SCORE, 0),
                    (T_SCORE + 0.05, 1),
                    (FOLD_T[0] - 0.03, 1),
                    (FOLD_T[0] + 0.05, 0),
                ]
            )
            + move(
                [
                    (0, (x0, y0)),
                    (T_SCORE, (x0, y0)),
                    (FOLD_T[0], (px(v), slot_y(i) + jit)),
                ]
            )
            + "</circle>"
        )

    for n, (t, s) in enumerate(
        (
            (T_INNER[0], "Optuna tunes on the inner folds"),
            (T_REFIT, "refit on the outer train folds"),
            (T_SCORE, "score the held-out test fold"),
        )
    ):
        y = 200 + 20 * n
        add(
            '<g opacity="0">'
            + op(appear(t, R2))
            + f'<circle class="b" cx="{f2(RX + 7)}" cy="{f2(y - 4)}" r="7"/>'
            + text(
                RX + 7,
                y - 0.4,
                str(n + 1),
                "wt",
                10,
                'font-weight="700" text-anchor="middle"',
            )
            + text(RX + 20, y, s, "fg", 10.5)
            + "</g>"
        )
    lx = RX
    for fill, label in (
        ('class="b"', "test"),
        ('class="b" fill-opacity=".2"', "train"),
        ('class="mu" fill-opacity=".6"', "validation"),
    ):
        add(rect(lx, 256, 12, 9, f'rx="2" {fill}'))
        add(text(lx + 16, 264.5, label, "mu", 10))
        lx += 16 + 5.6 * len(label) + 18
    add("</g>")
    return o


CAL_X0, CAL_Y1, CAL_S = 354.0, 236.0, 148.0
HIST_B, HIST_H = 282.0, 34.0
T_CAL = (T_COMPACT + 0.3, T_COMPACT + 2.85)
T_MORPH_A, T_MORPH_B = T_CAL[0] + 0.85, T_CAL[0] + 1.65
HIST_RAW = [31, 9, 5, 4, 3, 3, 4, 6, 11, 34]
HIST_CAL = [24, 12, 7, 5, 4, 4, 5, 8, 14, 27]


def build_calibration():
    """Return the SVG lines of the reliability diagram and its histogram."""
    o = ['<g opacity="0">' + op(window(*T_CAL))]
    add = o.append
    add(title(RX2, 70, "CALIBRATION"))
    add(text(X1, 70, BEST, "g", 11, 'font-weight="600" text-anchor="end"'))

    def cx(p):
        return CAL_X0 + p * CAL_S

    def cy(p):
        return CAL_Y1 - p * CAL_S

    add(
        f'<path class="ax" d="M{f2(cx(0))},{f2(cy(1))}V{f2(cy(0))}H{f2(cx(1))}'
        f'M{f2(cx(0))},{f2(HIST_B)}H{f2(cx(1))}" fill="none" stroke-width="1"/>'
    )
    add(
        f'<path class="ax" d="M{f2(cx(0))},{f2(cy(0))}L{f2(cx(1))},{f2(cy(1))}" '
        'stroke-width="1" stroke-dasharray="4 3" opacity=".7"/>'
    )
    add(text(cx(0), HIST_B + 13, "0", "mu", 10, 'text-anchor="middle"'))
    add(text(cx(1), HIST_B + 13, "1", "mu", 10, 'text-anchor="middle"'))
    add(text(cx(0) - 5, cy(1) + 4, "1", "mu", 10, 'text-anchor="end"'))
    add(text(cx(0) - 5, cy(0) + 4, "0", "mu", 10, 'text-anchor="end"'))
    add(
        text(
            cx(0.5),
            HIST_B + 27,
            "predicted probability",
            "mu",
            10,
            'text-anchor="middle"',
        )
    )
    for label, y in (("observed fraction", cy(0.5)), ("count", HIST_B - HIST_H / 2)):
        add(
            text(
                0,
                0,
                label,
                "mu",
                10,
                f'text-anchor="middle" transform="translate({f2(cx(0) - 16)},{f2(y)}) rotate(-90)"',
            )
        )

    ps = [0.05 + 0.1 * i for i in range(10)]
    raw = [
        1 / (1 + math.exp(-0.5 * math.log(p / (1 - p)))) + rng.gauss(0, 0.012)
        for p in ps
    ]
    cal = [min(max(p + rng.gauss(0, 0.012), 0.01), 0.99) for p in ps]

    def path(obs):
        pts = [f"{f2(cx(p))},{f2(cy(q))}" for p, q in zip(ps, obs, strict=True)]
        return "M" + "L".join(pts)

    def hist(counts):
        k = HIST_H / max(HIST_RAW + HIST_CAL)
        return "".join(
            f"M{f2(cx(i / 10) + 1.5)},{f2(HIST_B)}V{f2(HIST_B - c * k)}"
            f"H{f2(cx((i + 1) / 10) - 1.5)}V{f2(HIST_B)}Z"
            for i, c in enumerate(counts)
        )

    def morph(a, b):
        return [
            (0, a),
            (T_MORPH_A, a),
            (T_MORPH_B, b),
            (T_CAL[1] + 0.3, b),
            (T_CAL[1] + 0.35, a),
        ]

    fade_o = morph(1, 0)
    fade_g = [(t, 1 - v) for t, v in fade_o]
    d_raw, d_cal = path(raw), path(cal)
    h_raw, h_cal = hist(HIST_RAW), hist(HIST_CAL)
    for cls, fade in (("o", fade_o), ("g", fade_g)):
        add(f'<g opacity="{f2(fade[0][1])}">' + op(fade))
        add(
            f'<path class="{cls}" d="{h_raw}" fill-opacity=".55">'
            + anim("d", morph(h_raw, h_cal))
            + "</path>"
        )
        add(
            f'<path class="{cls}s" d="{d_raw}" fill="none" stroke-width="2" '
            'stroke-linejoin="round">' + anim("d", morph(d_raw, d_cal)) + "</path>"
        )
        for p, a, b in zip(ps, raw, cal, strict=True):
            add(
                f'<circle class="{cls}" cx="{f2(cx(p))}" cy="{f2(cy(a))}" r="2.7">'
                + anim("cy", morph(cy(a), cy(b)))
                + "</circle>"
            )
        add("</g>")

    lx = 540.0
    for cls, label, y, fade in (
        ("os", "raw", 106, fade_o),
        ("gs", "isotonic", 125, fade_g),
    ):
        add(
            f'<g opacity="{f2(0.45 + 0.55 * fade[0][1])}">'
            + op([(t, 0.45 + 0.55 * v) for t, v in fade])
            + f'<path class="{cls}" d="M{f2(lx)},{f2(y - 3.5)}h16" stroke-width="2"/>'
            + text(lx + 22, y, label, "fg", 11)
            + "</g>"
        )
    for label, y, v0, v1, full in (
        ("ECE", 172, 0.081, 0.019, 0.1),
        ("Brier", 224, 0.142, 0.118, 0.2),
    ):
        w0, w1 = v0 / full * 110, v1 / full * 110
        add(text(lx, y, label, "mu", 11, 'font-weight="600"'))
        for cls, fade in (("o", fade_o), ("g", fade_g)):
            add(
                animated(
                    rect(
                        lx,
                        y + 7,
                        w0,
                        9,
                        f'rx="2" class="{cls}" opacity="{f2(fade[0][1])}"',
                    ),
                    anim("width", morph(w0, w1)),
                    op(fade),
                )
            )
        add(text(lx + 118, y + 15, f"{v0:.3f}", "fg", 11))
        add(
            animated(
                text(
                    lx + 153,
                    y + 15,
                    f"→ {v1:.3f}",
                    "g",
                    11,
                    'font-weight="600" opacity="0"',
                ),
                op(fade_g),
            )
        )
    add("</g>")
    return o


SAMPLES = ["HG002", "NA12878", "REACH"]
XD = [[0.92, 0.81, 0.79], [0.83, 0.9, 0.84], [0.8, 0.85, 0.91]]
LODO = [0.84, 0.82, 0.86]
MX0, MY0, MCW, MCH = 384.0, 104.0, 52.0, 34.0
T_XD = (T_CAL[1] + 0.2, T_CAL[1] + 3.0)
SHIFT = [("coverage", 0.31), ("insert", 0.24), ("MAPQ", 0.17), ("split", 0.08)]
T_SHIFT = T_XD[0] + 1.9


def build_cross_dataset():
    """Return the SVG lines of the cross-dataset matrix and the feature shift."""
    o = ['<g opacity="0">' + op(window(*T_XD))]
    add = o.append
    add(title(RX2, 70, "CROSS-DATASET"))
    add(text(X1, 70, BEST, "g", 11, 'font-weight="600" text-anchor="end"'))
    add(
        text(
            MX0 + 1.5 * MCW,
            85,
            "test on",
            "mu",
            10,
            'text-anchor="middle" font-style="italic"',
        )
    )
    add(
        text(
            0,
            0,
            "train on",
            "mu",
            10,
            f'text-anchor="middle" font-style="italic" transform="translate({f2(RX2 + 8)},{f2(MY0 + 1.5 * MCH)}) rotate(-90)"',
        )
    )
    for j, s in enumerate(SAMPLES):
        add(text(MX0 + (j + 0.5) * MCW, 99, s, "fg", 10, 'text-anchor="middle"'))
        add(
            text(MX0 - 8, MY0 + (j + 0.5) * MCH + 4, s, "fg", 10.5, 'text-anchor="end"')
        )
    ly = MY0 + 3 * MCH + 10
    add(
        text(
            MX0 - 8,
            ly + MCH / 2 + 4,
            "LODO",
            "fg",
            10.5,
            'text-anchor="end" font-weight="700"',
        )
    )

    def alpha(v):
        return 0.1 + 0.85 * (v - 0.75) / 0.2

    order = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
    for n, (i, j) in enumerate(order):
        t = T_XD[0] + 0.3 + 0.1 * n + (0.15 if n >= 3 else 0)
        add(
            animated(
                rect(
                    MX0 + j * MCW + 1.5,
                    MY0 + i * MCH + 1.5,
                    MCW - 3,
                    MCH - 3,
                    f'rx="3" class="b" fill-opacity="{f2(alpha(XD[i][j]))}" opacity="0"',
                ),
                op(appear(t, R2, dur=0.15)),
            )
        )
    for j, v in enumerate(LODO):
        add(
            animated(
                rect(
                    MX0 + j * MCW + 1.5,
                    ly + 1.5,
                    MCW - 3,
                    MCH - 3,
                    f'rx="3" class="b bs" fill-opacity="{f2(alpha(v))}" stroke-width="1.2" '
                    'stroke-dasharray="3 2" opacity="0"',
                ),
                op(appear(T_XD[0] + 1.4 + 0.13 * j, R2, dur=0.15)),
            )
        )
    gx = MX0 + 3 * MCW + 12
    add(rect(gx, MY0, 8, 3 * MCH, 'rx="2" fill="url(#vt-scale)"'))
    add(
        text(
            0,
            0,
            "AUROC",
            "mu",
            10,
            f'text-anchor="middle" transform="translate({f2(gx + 22)},{f2(MY0 + 1.5 * MCH)}) rotate(-90)"',
        )
    )
    add(
        text(
            RX2,
            ly + MCH + 24,
            "LODO: each dataset scored by a model trained on the other two",
            "mu",
            10.5,
        )
    )

    kx = 604.0
    add('<g opacity="0">' + op(appear(T_SHIFT, R2)))
    add(
        rect(
            MX0 + 2 * MCW + 0.5,
            MY0 + 0.5,
            MCW - 1,
            MCH - 1,
            'rx="3.5" class="fgs" fill="none" stroke-width="1.6"',
        )
    )
    add(
        f'<path class="fgs" d="M{f2(MX0 + 3 * MCW)},{f2(MY0 + 8)}C{f2(kx - 30)},{f2(MY0 - 4)} '
        f'{f2(kx - 30)},{f2(MY0 - 4)} {f2(kx - 6)},{f2(MY0 - 4)}" fill="none" stroke-width="1" '
        'stroke-dasharray="3 2"/>'
    )
    add(text(kx, MY0, "shift, HG002 vs REACH", "fg", 10.5, 'font-weight="600"'))
    add("</g>")
    for i, (feat, ks) in enumerate(SHIFT):
        y = MY0 + 22 + 23 * i
        t = T_SHIFT + 0.1 + 0.1 * i
        w = ks * 220
        add(
            animated(
                text(kx, y + 8, feat, "fg", 10.5, 'opacity="0"'),
                op(appear(t, R2)),
            )
        )
        add(
            animated(
                rect(kx + 56, y, 0, 9, 'rx="2" class="o"'),
                grow("width", t, t + 0.35, 0, w, R2),
            )
        )
        add(
            animated(
                text(kx + 62 + w, y + 8.5, f"{ks:.2f}", "fg", 10, 'opacity="0"'),
                op(appear(t + 0.3, R2)),
            )
        )
    add(
        animated(
            text(
                kx,
                MY0 + 22 + 23 * len(SHIFT) + 8,
                "KS statistic per feature",
                "mu",
                10,
                'opacity="0"',
            ),
            op(appear(T_SHIFT + 0.5, R2)),
        )
    )
    add("</g>")
    return o


PB_X, PB_W = RX2, 262.0
TAU = 0.62
SHAP = [
    ("split_ratio", 0.92),
    ("discordant_ratio", 0.66),
    ("coverage_inside", 0.48),
    ("SVLEN_CALLER", -0.16),
]
T_VERDICT = T_XD[1] + 0.15
CSV_X, CSV_Y, CSV_W, CSV_H = 528.0, 214.0, 228.0, 92.0
CSV_ROWS = [
    ("row_id", "prob_class_1", "predicted_class"),
    ("call_A", "0.94", "1"),
    ("call_B", "0.07", "0"),
]
T_CSV = T_VERDICT + 1.4


def build_verdict():
    """Return the SVG lines of the verdict card and predictions.csv."""
    o = [
        '<g opacity="0">'
        + op([(0, 0), (T_VERDICT, 0), (T_VERDICT + 0.25, 1), (R2 - 0.05, 1), (R2, 0)])
    ]
    add = o.append
    add(title(RX2, 70, "VERDICT"))
    add(
        text(
            X1,
            70,
            f"{BEST}, calibrated",
            "g",
            11,
            'font-weight="600" text-anchor="end"',
        )
    )

    calls = [
        ("call A", "MANTA + SMOOVE", 0.94, "TRUE", "g", 100.0, T_VERDICT + 0.2),
        ("call B", "DELLY only", 0.07, "FALSE", "o", 156.0, T_VERDICT + 0.45),
    ]
    for name, support, p, verdict, cls, y, t in calls:
        add(text(RX2, y, name, "fg", 11.5, 'font-weight="600"'))
        add(rect(RX2 + 46, y - 10, 28, 13, 'rx="3" class="b"'))
        add(
            text(RX2 + 60, y, "DEL", "wt", 10, 'font-weight="700" text-anchor="middle"')
        )
        add(text(RX2 + 82, y, support, "mu", 11))
        by = y + 10
        add(rect(PB_X, by, PB_W, 10, 'rx="5" class="mu" fill-opacity=".18"'))
        add(
            animated(
                rect(PB_X, by, 0, 10, f'rx="5" class="{cls}"'),
                grow("width", t, t + 0.5, 0, max(p * PB_W, 10), R2),
            )
        )
        add(
            animated(
                text(
                    PB_X + PB_W + 10,
                    by + 9.5,
                    f"{p:.2f}",
                    "fg",
                    12,
                    'font-weight="700" opacity="0"',
                ),
                op(appear(t + 0.45, R2)),
            )
        )
        add(
            popped(
                X1 - 64,
                by + 5,
                t + 0.5,
                R2,
                rect(-32, -11, 64, 22, f'rx="11" fill="{FILL[cls]}"')
                + text(
                    0, 4.4, verdict, "wt", 12, 'font-weight="700" text-anchor="middle"'
                ),
            )
        )
    tx = PB_X + TAU * PB_W
    ticks = "".join(f"M{f2(tx)},{f2(y + 7)}v16" for *_, y, _ in calls)
    add(
        '<g opacity="0">'
        + op(appear(T_VERDICT + 0.1, R2))
        + f'<path class="fgs" d="{ticks}" stroke-width="1.5"/>'
        + text(tx, 196, "τ, Youden J", "mu", 10, 'text-anchor="middle"')
        + "</g>"
    )

    zx, sy = 430.0, 238.0
    add(text(RX2, 226, "SHAP attribution, call A", "mu", 10.5, 'font-weight="600"'))
    add(
        f'<path class="ax" d="M{f2(zx)},{f2(sy - 4)}V{f2(sy + 4 * 17)}" stroke-width="1"/>'
    )
    for i, (feat, v) in enumerate(SHAP):
        y = sy + i * 17
        t = T_VERDICT + 0.9 + 0.12 * i
        w = abs(v) * 76
        add(text(RX2, y + 8, feat, "fg", 10.5))
        bar = rect(zx, y, 0, 10, f'rx="1.5" class="{"g" if v > 0 else "o"}"')
        if v > 0:
            add(animated(bar, grow("width", t, t + 0.35, 0, w, R2)))
        else:
            add(
                animated(
                    bar,
                    grow("width", t, t + 0.35, 0, w, R2),
                    grow("x", t, t + 0.35, zx, zx - w, R2),
                )
            )

    add(
        '<g opacity="0">'
        + op(appear(T_CSV, R2))
        + rect(
            CSV_X,
            CSV_Y,
            CSV_W,
            CSV_H,
            'rx="6" class="mu fgs" fill-opacity=".07" stroke-opacity=".35"',
        )
        + text(
            CSV_X + 10,
            CSV_Y + 17,
            "predictions.csv",
            "fg",
            10.5,
            f'font-weight="700" {MONO}',
        )
        + text(
            CSV_X + CSV_W - 10,
            CSV_Y + 17,
            "predict --proba",
            "mu",
            10,
            f'text-anchor="end" {MONO}',
        )
        + "</g>"
    )
    cols = [CSV_X + 10, CSV_X + 54, CSV_X + 136]
    for n, row in enumerate(CSV_ROWS):
        y = CSV_Y + 42 + 18 * n
        t = T_CSV + 0.2 + 0.4 * n
        add(
            f'<defs><clipPath id="vt-csv{n}">'
            + animated(
                rect(CSV_X, y - 11, 0, 15),
                grow("width", t, t + 0.35, 0, CSV_W, R2),
            )
            + "</clipPath></defs>"
        )
        cls = "mu" if n == 0 else "fg"
        cells = "".join(
            text(x, y, s, cls, 10, f'{MONO} font-weight="{600 if n else 400}"')
            for x, s in zip(cols, row, strict=True)
        )
        add(f'<g clip-path="url(#vt-csv{n})">{cells}</g>')
    add("</g>")
    return o


def build() -> str:
    """Return the SVG document as a string."""
    o = []
    add = o.append

    add('<?xml version="1.0" encoding="UTF-8"?>')
    add(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">'
    )
    add('<title id="t">VarTrustML workflow</title>')
    add(
        '<desc id="d">Animated loop with synthetic, illustrative values. Three '
        "short-read structural variant callers scan a stretch of chr1: MANTA and "
        "SMOOVE report deletion call A, only DELLY reports call B. The read pileup "
        "and its coverage track show split reads, discordant pairs, soft clips and "
        "a coverage drop at call A, a lens shows the soft-clipped bases at one "
        "breakpoint, and the evidence becomes a feature vector. Six models "
        "(CatBoost, XGBoost, Random Forest, MLP, Logistic Regression, KNN) are "
        "compared under nested cross-validation against the callers with McNemar "
        "and DeLong tests and Holm correction, and ranked. The best model is "
        "calibrated, tested on every train and test pair of HG002, NA12878 and "
        "REACH and with leave-one-dataset-out, next to the feature shift between "
        "datasets, and its calibrated probability, against the Youden threshold, "
        "marks call A true and call B false, with SHAP attributions and the rows "
        "written to predictions.csv.</desc>"
    )
    # mid-tones legible on white and on #0d1117, the dark block only nudges them
    add(
        "<style>"
        'text{font-family:system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif}'
        ".fg{fill:#6e7781}.fgs{stroke:#6e7781}.mu{fill:#7d8590}.ax{stroke:#7d8590}"
        ".wt{fill:#ffffff}"
        f".b{{fill:{BLUE}}}.bs{{stroke:{BLUE}}}.bst{{stop-color:{BLUE}}}"
        f".g{{fill:{GREEN}}}.gs{{stroke:{GREEN}}}"
        f".o{{fill:{ORANGE}}}.os{{stroke:{ORANGE}}}"
        f".nA{{fill:{GREEN}}}.nC{{fill:{BLUE}}}.nG{{fill:{ORANGE}}}.nT{{fill:{RED}}}"
        f".pill{{fill:{BLUE};fill-opacity:.12;stroke:{BLUE};stroke-opacity:.7}}"
        "@media (prefers-color-scheme:dark){"
        f".fg{{fill:#8b949e}}.fgs{{stroke:#8b949e}}.b{{fill:{BLUE_DK}}}.bs{{stroke:{BLUE_DK}}}"
        f".bst{{stop-color:{BLUE_DK}}}.o{{fill:{ORANGE_DK}}}.os{{stroke:{ORANGE_DK}}}"
        f".nC{{fill:{BLUE_DK}}}.nG{{fill:{ORANGE_DK}}}"
        f".pill{{fill:{BLUE_DK};stroke:{BLUE_DK}}}"
        "}"
        "</style>"
    )
    add(
        "<defs>"
        '<linearGradient id="vt-scale" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" class="bst" stop-opacity=".95"/>'
        '<stop offset="1" class="bst" stop-opacity=".1"/>'
        "</linearGradient>"
        '<linearGradient id="vt-trail" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0" class="bst" stop-opacity="0"/>'
        '<stop offset="1" class="bst" stop-opacity=".3"/>'
        "</linearGradient>"
        "</defs>"
    )

    n = len(STAGES)
    step_w = (X1 - X0) / n
    centers = [X0 + (k + 0.5) * step_w for k in range(n)]
    pill_w, pill_h, pill_cy = 90.0, 24.0, 28.0
    pk = [(0, 0)]
    for k in range(n - 1):
        pk += [
            (SW[k] - 0.2, centers[k] - centers[0]),
            (SW[k] + 0.25, centers[k + 1] - centers[0]),
        ]
    pk += [(SW[-1] - 0.05, centers[-1] - centers[0]), (D - 0.04, 0)]
    add(
        animated(
            rect(
                centers[0] - pill_w / 2,
                pill_cy - pill_h / 2,
                pill_w,
                pill_h,
                f'class="pill" rx="{f2(pill_h / 2)}"',
            ),
            move([(t, (dx, 0)) for t, dx in pk]),
        )
    )
    for k in range(1, n):
        add(
            text(
                X0 + k * step_w,
                pill_cy + 4.5,
                "›",
                "mu",
                13,
                'text-anchor="middle" opacity=".55"',
            )
        )
    for k, (short, _, _) in enumerate(STAGES):
        on = stage_opacity(k)
        off = [(t, 1 - v) for t, v in on]
        label = f'<tspan font-size="10.5" opacity=".7">{k + 1}</tspan><tspan dx="4">{short}</tspan>'
        common = (
            f'x="{f2(centers[k])}" y="{f2(pill_cy + 4.5)}" font-size="12.5" '
            'font-weight="600" text-anchor="middle"'
        )
        add(
            f'<text class="mu" {common} opacity="{f2(off[0][1])}">{label}{op(off)}</text>'
        )
        add(f'<text class="b" {common} opacity="{f2(on[0][1])}">{label}{op(on)}</text>')

    o += build_act1()

    add('<g opacity="0">' + op(ACT2))
    scores = fold_scores()
    o += build_compare(scores)
    o += build_nested_cv(scores)
    o += build_calibration()
    o += build_cross_dataset()
    o += build_verdict()
    add("</g>")

    for k, (_, name, desc) in enumerate(STAGES):
        on = stage_opacity(k)
        add(
            f'<text x="{f2((X0 + X1) / 2)}" y="369" font-size="13.5" text-anchor="middle" '
            f'opacity="{f2(on[0][1])}">'
            f'<tspan class="fg" font-weight="600">{name}</tspan>'
            f'<tspan class="mu" dx="8">{desc}</tspan>{op(on)}</text>'
        )
    add(
        text(
            (X0 + X1) / 2,
            386,
            "synthetic values, for illustration only",
            "mu",
            10,
            'text-anchor="middle" font-style="italic"',
        )
    )

    add("</svg>")
    return "\n".join(o) + "\n"


def main() -> None:
    """Write the SVG to the path given on the command line, or to DEFAULT_OUT."""
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    svg = build()
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(svg)
    print(f"Wrote {out} ({len(svg.encode('utf-8'))} bytes)")


if __name__ == "__main__":
    main()
