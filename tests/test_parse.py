from pathlib import Path

from qd_suite.data.parse_ndjson import iter_sketches, parse_line


def test_parse_line_roundtrip(tmp_path: Path):
    line = (
        '{"key_id":"1","word":"cat","drawing":[[[0,1],[0,1],[0,1]]],'
        '"extra":"meta"}'
    )
    sketch = parse_line(line)
    assert sketch.word == "cat"
    assert len(sketch.strokes) == 1
    assert sketch.strokes[0][0] == (0.0, 0.0, 0.0)

    ndjson = tmp_path / "cat.ndjson"
    ndjson.write_text(line + "\n", encoding="utf-8")
    sketches = list(iter_sketches(ndjson))
    assert sketches[0].key_id == "1"
