from qd_suite.data.canonicalize import to_pointseq
from qd_suite.data.parse_ndjson import RawSketch
from qd_suite.repr.normalize import normalize_time, normalize_xy


def test_canonicalize_and_normalize_time():
    raw = RawSketch(
        key_id="1",
        word="cat",
        strokes=[[(0, 0, 0), (1, 1, 2)], [(1, 1, 3), (2, 2, 4)]],
        metadata={},
    )
    seq = to_pointseq(raw)
    assert len(seq.points) == 5  # includes a separator
    seq_time = normalize_time(seq, mode="unit")
    assert seq_time.points[0].t == 0.0
    assert seq_time.points[-1].t == 1.0


def test_normalize_xy_bounds():
    raw = RawSketch(key_id="1", word="cat", strokes=[[(0, 0, 0), (2, 2, 1)]], metadata={})
    seq = to_pointseq(raw, normalize_time_mode=None)
    seq_norm = normalize_xy(seq)
    xs = [p.x for p in seq_norm.points if p.pen.name == "DOWN"]
    assert max(xs) <= 1.0 + 1e-6
    assert min(xs) >= -1.0 - 1e-6
