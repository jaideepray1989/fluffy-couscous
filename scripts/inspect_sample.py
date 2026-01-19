#!/usr/bin/env python3
import argparse
from pathlib import Path

from qd_suite.data.parse_ndjson import iter_sketches


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a QuickDraw ndjson sample.")
    parser.add_argument("path", help="Path to ndjson file")
    parser.add_argument("--limit", type=int, default=1, help="How many sketches to print")
    args = parser.parse_args(argv)

    count = 0
    for sketch in iter_sketches(Path(args.path)):
        print(f"key_id={sketch.key_id}, word={sketch.word}, strokes={len(sketch.strokes)}")
        for idx, stroke in enumerate(sketch.strokes):
            print(f"  stroke {idx}: {len(stroke)} points, t in [{stroke[0][2]}, {stroke[-1][2]}]")
        count += 1
        if count >= args.limit:
            break
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
