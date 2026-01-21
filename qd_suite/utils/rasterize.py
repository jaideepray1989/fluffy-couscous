from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from qd_suite.repr.pointseq import PenState, PointSequence


def rasterize(seq: PointSequence, size: int = 64, stroke_width: int = 2, padding: int = 2) -> Image.Image:
    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)
    strokes = seq.strokes()
    if not strokes:
        return img
    coords = []
    for stroke in strokes:
        coords.extend([(p.x, p.y) for p in stroke])
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    if not xs or not ys:
        return img
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    scale = max(max_x - min_x, max_y - min_y)
    scale = scale if scale > 0 else 1.0
    for stroke in strokes:
        pts = []
        for p in stroke:
            nx = (p.x - min_x) / scale
            ny = (p.y - min_y) / scale
            px = padding + nx * (size - 2 * padding)
            py = padding + ny * (size - 2 * padding)
            pts.append((px, py))
        if len(pts) >= 2:
            draw.line(pts, fill=0, width=stroke_width)
        elif len(pts) == 1:
            draw.ellipse([pts[0][0] - stroke_width, pts[0][1] - stroke_width, pts[0][0] + stroke_width, pts[0][1] + stroke_width], fill=0)
    return img
