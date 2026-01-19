from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List
from urllib.request import urlretrieve


GCS_URI = "gs://quickdraw_dataset/full/raw/{cls}.ndjson"
HTTPS_URI = "https://storage.googleapis.com/quickdraw_dataset/full/raw/{cls}.ndjson"


def download_gsutil(classes: Iterable[str], out_dir: Path, gsutil_bin: str = "gsutil") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for cls in classes:
        uri = GCS_URI.format(cls=cls)
        dest = out_dir / f"{cls}.ndjson"
        cmd = [gsutil_bin, "-m", "cp", uri, str(dest)]
        subprocess.run(cmd, check=True)


def download_https(classes: Iterable[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for cls in classes:
        uri = HTTPS_URI.format(cls=cls)
        dest = out_dir / f"{cls}.ndjson"
        urlretrieve(uri, dest)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download QuickDraw raw ndjson files.")
    parser.add_argument("--classes", required=True, help="Comma-separated class names")
    parser.add_argument("--out", required=True, help="Output directory for ndjson")
    parser.add_argument("--method", choices=["gsutil", "https"], default="https", help="Download method")
    parser.add_argument("--gsutil-bin", default="gsutil", help="Path to gsutil (if using gsutil)")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    out_dir = Path(args.out)
    try:
        if args.method == "gsutil":
            download_gsutil(classes, out_dir, gsutil_bin=args.gsutil_bin)
        else:
            download_https(classes, out_dir)
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
