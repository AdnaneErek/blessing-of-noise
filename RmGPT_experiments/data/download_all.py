# RmGPT/data/download_all.py
from __future__ import annotations
import subprocess
import sys

SCRIPTS = [
    "RmGPT.data.download_cwru",
    "RmGPT.data.download_sliet",
    "RmGPT.data.download_qpzz2",
    "RmGPT.data.download_smu",
    "RmGPT.data.download_xjtu",
]

def main() -> None:
    args = sys.argv[1:]
    for mod in SCRIPTS:
        cmd = [sys.executable, "-m", mod] + args
        print("\n" + "=" * 80)
        print("Running:", " ".join(cmd))
        print("=" * 80)
        subprocess.check_call(cmd)

if __name__ == "__main__":
    main()
