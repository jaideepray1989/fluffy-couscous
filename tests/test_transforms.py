from qd_suite.repr.pointseq import flatten_strokes
from qd_suite.transforms import dropout, prefix, resample, simplify


def sample_seq():
    strokes = [
        [(0, 0, 0), (1, 0, 1), (2, 0, 2)],
        [(2, 0, 3), (2, 1, 4)],
    ]
    return flatten_strokes(strokes, metadata={"word": "cat"})


def test_prefix_points_and_strokes():
    seq = sample_seq()
    sub = prefix.prefix_points(seq, 50)
    assert len(sub.points) > 0
    sub_stroke = prefix.prefix_strokes(seq, 50)
    assert len(sub_stroke.strokes()) == 1


def test_dropout_and_resample():
    seq = sample_seq()
    dropped = dropout.point_dropout(seq, 0.5, seed=0)
    assert dropped.strokes()
    resampled = resample.resample_uniform(seq, 4)
    assert all(len(s) == 4 for s in resampled.strokes())


def test_simplify():
    seq = sample_seq()
    simple = simplify.simplify_rdp(seq, epsilon=0.1)
    assert len(simple.strokes()[0]) >= 2
