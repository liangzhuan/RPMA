"""Download and place COIL-20 and Cropped Extended Yale B datasets.

Run from the project root, for example:
    python scripts/download_image_datasets.py --dataset coil20
    python scripts/download_image_datasets.py --dataset yaleb
    python scripts/download_image_datasets.py --dataset both

The script places files in:
    datasets/data/coil20/
    datasets/data/CroppedYale/

Notes:
- COIL-20 uses the Columbia CAVE official processed COIL-20 archive.
- Cropped Yale B first tries the legacy UCSD URL. If it is no longer reachable,
  rerun with --allow-yale-mirror to use a public mirror.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "datasets" / "data"
CACHE_DIR = PROJECT_ROOT / "datasets" / "_downloads"

COIL20_URLS = [
    "https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip",
    "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip",
]

YALEB_OFFICIAL_URLS = [
    "https://vision.ucsd.edu/extyaleb/CroppedYaleBZip/CroppedYale.zip",
    "http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/CroppedYale.zip",
]

# Use only when the legacy UCSD URL is down.
YALEB_MIRROR_URLS = [
    "https://raw.githubusercontent.com/trokas/ai_primer/master/img/CroppedYale.zip",
]


def _download_with_fallback(urls: list[str], out_path: Path) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in urls:
        try:
            print(f"Downloading: {url}")
            with urllib.request.urlopen(url, timeout=60) as response:
                total = int(response.headers.get("Content-Length") or 0)
                downloaded = 0
                chunk_size = 1024 * 1024
                with out_path.open("wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            print(f"  {downloaded/1024/1024:.1f}/{total/1024/1024:.1f} MB ({pct:.1f}%)", end="\r")
            print(f"\nSaved: {out_path}")
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"Failed: {url}\n  {exc}")
            if out_path.exists():
                out_path.unlink()
    raise RuntimeError(f"All download URLs failed. Last error: {last_error}")


def _safe_extract_zip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target = dest / member.filename
            if not str(target.resolve()).startswith(str(dest.resolve())):
                raise RuntimeError(f"Unsafe zip member: {member.filename}")
        zf.extractall(dest)


def download_coil20(force: bool = False) -> None:
    target = DATA_ROOT / "coil20"
    existing = list(target.rglob("obj*__*.png"))
    if existing and not force:
        print(f"COIL-20 already found: {len(existing)} images under {target}")
        return
    if force and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    zip_path = CACHE_DIR / "coil-20-proc.zip"
    if force or not zip_path.exists():
        _download_with_fallback(COIL20_URLS, zip_path)

    tmp_extract = CACHE_DIR / "coil20_extract"
    if tmp_extract.exists():
        shutil.rmtree(tmp_extract)
    _safe_extract_zip(zip_path, tmp_extract)

    images = sorted(tmp_extract.rglob("obj*__*.png"))
    if not images:
        raise RuntimeError("Downloaded COIL-20 archive did not contain obj*__*.png files.")

    for img in images:
        shutil.copy2(img, target / img.name)

    print(f"Placed COIL-20: {len(images)} images -> {target}")


def download_yaleb(force: bool = False, allow_mirror: bool = False) -> None:
    target = DATA_ROOT / "CroppedYale"
    existing = list(target.rglob("*.pgm"))
    if existing and not force:
        print(f"Cropped Yale B already found: {len(existing)} PGM files under {target}")
        return
    if force and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    zip_path = CACHE_DIR / "CroppedYale.zip"
    urls = list(YALEB_OFFICIAL_URLS)
    if allow_mirror:
        urls += YALEB_MIRROR_URLS
    if force or not zip_path.exists():
        _download_with_fallback(urls, zip_path)

    tmp_extract = CACHE_DIR / "yaleb_extract"
    if tmp_extract.exists():
        shutil.rmtree(tmp_extract)
    _safe_extract_zip(zip_path, tmp_extract)

    extracted_root = tmp_extract / "CroppedYale"
    if extracted_root.exists():
        # Copy contents of CroppedYale/ into datasets/data/CroppedYale/
        for item in extracted_root.iterdir():
            dst = target / item.name
            if dst.exists():
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            if item.is_dir():
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)
    else:
        # Some mirrors may extract directly into subject folders.
        for item in tmp_extract.iterdir():
            dst = target / item.name
            if item == target:
                continue
            if dst.exists():
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            if item.is_dir():
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)

    pgms = list(target.rglob("*.pgm"))
    if not pgms:
        raise RuntimeError("Cropped Yale B archive extracted, but no .pgm images were found.")
    print(f"Placed Cropped Yale B: {len(pgms)} PGM files -> {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["coil20", "yaleb", "both"], default="both")
    parser.add_argument("--force", action="store_true", help="Redownload and overwrite existing extracted data.")
    parser.add_argument(
        "--allow-yale-mirror",
        action="store_true",
        help="If the legacy UCSD Yale B URL fails, allow a public mirror fallback.",
    )
    args = parser.parse_args()

    if args.dataset in {"coil20", "both"}:
        download_coil20(force=args.force)
    if args.dataset in {"yaleb", "both"}:
        download_yaleb(force=args.force, allow_mirror=args.allow_yale_mirror)

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
