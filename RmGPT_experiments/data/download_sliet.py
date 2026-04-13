# RmGPT/data/download_sliet.py
from __future__ import annotations
from pathlib import Path

from RmGPT.data.download_utils import DownloadSpec, build_argparser, fetch_or_manual

DATASET = "sliet"

def main() -> None:
    ap = build_argparser(DATASET)
    args = ap.parse_args()
    data_root = Path(args.data_root)

    spec = DownloadSpec(
        name="SLIET Bearing Dataset",
        url=None,  # often requires manual access
        filename="sliet.zip",
        sha256=None,
        extract=True,
    )
    fetch_or_manual(spec, data_root=data_root, dataset_name=DATASET, force=args.force)

if __name__ == "__main__":
    main()
