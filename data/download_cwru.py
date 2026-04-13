# RmGPT/data/download_cwru.py
from __future__ import annotations
from pathlib import Path

from RmGPT.data.download_utils import DownloadSpec, build_argparser, fetch_or_manual

DATASET = "cwru"

def main() -> None:
    ap = build_argparser(DATASET)
    args = ap.parse_args()
    data_root = Path(args.data_root)

    # NOTE:
    # Many public mirrors exist; URLs can break. Keep url=None (manual) for robust reproduction.
    spec = DownloadSpec(
        name="CWRU Bearing Dataset",
        url=None,  # set to a stable direct download if your lab has one
        filename="cwru.zip",  # whatever archive you will place under data/raw/cwru/manual/
        sha256=None,          # fill if you want strict verification
        extract=True,
    )

    fetch_or_manual(spec, data_root=data_root, dataset_name=DATASET, force=args.force)

if __name__ == "__main__":
    main()
