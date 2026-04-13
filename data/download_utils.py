# RmGPT/data/download_utils.py
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request

CHUNK_SIZE = 1024 * 1024  # 1MB


@dataclass
class DownloadSpec:
    name: str
    url: Optional[str]                 # None -> manual only
    filename: str                      # local archive name
    sha256: Optional[str] = None       # optional checksum
    extract: bool = True
    # where archive will be stored:
    # data/raw/<dataset>/archives/<filename>
    # extracted to:
    # data/raw/<dataset>/extracted/


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def download_to(url: str, out_path: Path, user_agent: str = "Mozilla/5.0") -> None:
    ensure_dir(out_path.parent)
    req = Request(url, headers={"User-Agent": user_agent})
    with urlopen(req) as r, out_path.open("wb") as f:
        total = r.headers.get("Content-Length")
        total = int(total) if total is not None else None
        downloaded = 0
        while True:
            chunk = r.read(CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = 100.0 * downloaded / total
                print(f"\r  downloaded {downloaded}/{total} bytes ({pct:.1f}%)", end="")
        print("")


def extract_archive(archive_path: Path, out_dir: Path) -> None:
    ensure_dir(out_dir)
    suffix = "".join(archive_path.suffixes).lower()

    if suffix.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(out_dir)
    elif suffix.endswith(".tar.gz") or suffix.endswith(".tgz") or suffix.endswith(".tar"):
        mode = "r:gz" if suffix.endswith((".tar.gz", ".tgz")) else "r:"
        with tarfile.open(archive_path, mode) as t:
            t.extractall(out_dir)
    else:
        raise ValueError(f"Unsupported archive type: {archive_path.name}")


def dataset_paths(data_root: Path, dataset_name: str) -> dict[str, Path]:
    base = data_root / "raw" / dataset_name
    paths = {
        "base": base,
        "archives": base / "archives",
        "extracted": base / "extracted",
        "manual": base / "manual",
    }
    for p in paths.values():
        ensure_dir(p)
    return paths


def fetch_or_manual(spec: DownloadSpec, data_root: Path, dataset_name: str, force: bool = False) -> Path:
    paths = dataset_paths(data_root, dataset_name)
    archive_path = paths["archives"] / spec.filename

    if archive_path.exists() and not force:
        print(f"[{spec.name}] Archive already exists: {archive_path}")
    else:
        if spec.url is None:
            # manual mode
            manual_candidate = paths["manual"] / spec.filename
            if not manual_candidate.exists():
                raise FileNotFoundError(
                    f"[{spec.name}] No URL provided (manual-only). Please place the file at:\n"
                    f"  {manual_candidate}\n"
                    f"Then re-run."
                )
            ensure_dir(paths["archives"])
            shutil.copy2(manual_candidate, archive_path)
            print(f"[{spec.name}] Copied manual file -> {archive_path}")
        else:
            print(f"[{spec.name}] Downloading from: {spec.url}")
            download_to(spec.url, archive_path)

    if spec.sha256:
        got = sha256_file(archive_path)
        if got.lower() != spec.sha256.lower():
            raise RuntimeError(
                f"[{spec.name}] SHA256 mismatch!\nExpected: {spec.sha256}\nGot:      {got}\nFile: {archive_path}"
            )
        print(f"[{spec.name}] SHA256 verified.")

    if spec.extract:
        out_dir = paths["extracted"]
        marker = out_dir / ".extracted_ok"
        if marker.exists() and not force:
            print(f"[{spec.name}] Already extracted: {out_dir}")
        else:
            print(f"[{spec.name}] Extracting to: {out_dir}")
            # clear previous extracted if force
            if out_dir.exists() and force:
                shutil.rmtree(out_dir)
                ensure_dir(out_dir)
            extract_archive(archive_path, out_dir)
            marker.write_text("ok\n")
            print(f"[{spec.name}] Extraction complete.")

    return archive_path


def build_argparser(dataset_name: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(f"Download {dataset_name} dataset")
    p.add_argument("--data_root", type=str, default="data", help="Project data root (default: data)")
    p.add_argument("--force", action="store_true", help="Re-download / re-extract")
    return p
