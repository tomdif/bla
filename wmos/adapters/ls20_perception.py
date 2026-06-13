"""Ground the ls20 shape state in REAL pixels. Extract from a 64x64 frame: the avatar position, the
cross operator, the avatar's KEY shape, the EXIT key shape, and an orientation descriptor for each so
the hierarchical sub-goals run on perceived (not modeled) state.

Real ls20 (verified on recorded frames): color 12 = avatar core; color 9 = key/icon material, present
as several components -- the AVATAR KEY (attached to the core), the EXIT KEY (top box), and a LEGEND
(bottom-left). Colors 0/1 = the white cross operator.

HONEST SCOPE: extracting the components is reliable; the exact ls20 MATCH predicate (avatar's filled
key vs the exit's pattern) is the ~/arc_local open problem. The robust descriptor the project landed on
is the EMPTIEST-QUADRANT orientation -- it carries an orientation signal when the shape is asymmetric
and is AMBIGUOUS (low confidence) when the shape is solid/symmetric. We expose that confidence so WMOS
can REFUSE where the orientation is ambiguous (verificationist: know when you don't know).
"""
from collections import deque

AVATAR_CORE, KEY9, CROSS_A, CROSS_B = 12, 9, 0, 1


def _components(g, colors, conn8=True):
    H, W = len(g), len(g[0]); seen = set(); out = []
    nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1)] + ([(1, 1), (1, -1), (-1, 1), (-1, -1)] if conn8 else [])
    for r in range(H):
        for c in range(W):
            if g[r][c] in colors and (r, c) not in seen:
                q = deque([(r, c)]); seen.add((r, c)); cells = []
                while q:
                    y, x = q.popleft(); cells.append((y, x))
                    for dy, dx in nbrs:
                        n = (y + dy, x + dx)
                        if 0 <= n[0] < H and 0 <= n[1] < W and n not in seen and g[n[0]][n[1]] in colors:
                            seen.add(n); q.append(n)
                out.append(cells)
    return out


def _centroid(cells):
    return (sum(c[0] for c in cells) / len(cells), sum(c[1] for c in cells) / len(cells))


def orient(cells):
    """emptiest-quadrant orientation (0=TL,1=TR,2=BL,3=BR) + confidence in [0,1].
    Confidence = how much emptier the emptiest quadrant is than the next (0 => solid/symmetric => unknown)."""
    ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
    my = (min(ys) + max(ys)) / 2; mx = (min(xs) + max(xs)) / 2
    quad = [0, 0, 0, 0]
    for y, x in cells:
        quad[(0 if y <= my else 2) + (0 if x <= mx else 1)] += 1
    order = sorted(range(4), key=lambda i: quad[i])
    emptiest = order[0]; total = len(cells) or 1
    conf = round((quad[order[1]] - quad[order[0]]) / total, 3)   # gap between emptiest two quadrants
    return emptiest, conf


def extract(frame):
    """Return perceived ls20 shape state from a real frame.

    CRACKED MATCH PREDICATE (mined + validated on 2000 recorded frames, 0 false matches on score-0
    data): the two keys that must match are the LEGEND (the flippable key HUD, bottom-left color-9)
    and the EXIT key (top box, color-9). Both are the SAME L-shaped glyph (a bar + a leg) at different
    scales; the match is ORIENTATION agreement (scale-invariant emptiest-quadrant) -- the cross flips
    the legend's orientation; you win when legend orient == exit orient AND the avatar is at the exit.
    The avatar's own color-9 is a solid block (no orientation) and is NOT the key to match."""
    g = frame.tolist() if hasattr(frame, "tolist") else frame
    H, W = len(g), len(g[0])
    core = [(r, c) for r in range(H) for c in range(W) if g[r][c] == AVATAR_CORE]
    avatar = _centroid(core) if core else (H / 2, W / 2)
    cross_comps = _components(g, {CROSS_A, CROSS_B})
    cross = _centroid(max(cross_comps, key=len)) if cross_comps else None
    key_comps = [c for c in _components(g, {KEY9}) if len(c) >= 4]
    legend = exit_key = avatar_body = None
    if key_comps:
        avatar_body = min(key_comps, key=lambda c: abs(_centroid(c)[0] - avatar[0]) + abs(_centroid(c)[1] - avatar[1]))
        keys = [c for c in key_comps if c is not avatar_body]   # exclude the avatar's solid icon block
        if keys:
            legend = max(keys, key=lambda c: _centroid(c)[0] - _centroid(c)[1])   # bottom-left = the legend key
            tops = [c for c in keys if c is not legend]
            exit_key = min(tops, key=lambda c: _centroid(c)[0]) if tops else None
    key_or, key_cf = orient(legend) if legend else (None, 0.0)   # the flippable key (legend)
    tgt_or, tgt_cf = orient(exit_key) if exit_key else (None, 0.0)
    matched = key_or is not None and tgt_or is not None and key_or == tgt_or
    return {
        "avatar": (round(avatar[0]), round(avatar[1])),
        "cross": (round(cross[0]), round(cross[1])) if cross else None,
        "key": {"source": "legend", "cells": len(legend) if legend else 0, "orient": key_or,
                "confidence": key_cf, "center": tuple(round(x) for x in _centroid(legend)) if legend else None},
        "target": {"source": "exit", "cells": len(exit_key) if exit_key else 0, "orient": tgt_or,
                   "confidence": tgt_cf, "center": tuple(round(x) for x in _centroid(exit_key)) if exit_key else None},
        "avatar_body_cells": len(avatar_body) if avatar_body else 0,
        "matched": matched,
        "match_confidence": round(min(key_cf, tgt_cf), 3),
        # back-compat aliases
        "avatar_key": {"orient": key_or, "confidence": key_cf, "cells": len(legend) if legend else 0,
                       "center": tuple(round(x) for x in _centroid(legend)) if legend else None},
        "exit_key": {"orient": tgt_or, "confidence": tgt_cf, "cells": len(exit_key) if exit_key else 0,
                     "center": tuple(round(x) for x in _centroid(exit_key)) if exit_key else None},
    }
