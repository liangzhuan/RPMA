"""Download/install Cropped Extended Yale B into datasets/data/yaleB.

Recommended layout produced:

    Community_detection/
    └─ datasets/data/
       ├─ coil20/
       └─ yaleB/
          ├─ yaleB01/*.{pgm,png,jpg,...}
          ├─ yaleB02/*.{pgm,png,jpg,...}
          └─ ...

Why v2:
- The old UCSD CroppedYale.zip legacy URL is often 404.
- This script supports three practical routes:
    1) --zip-path: install a manually downloaded CroppedYale archive.
    2) --source kaggle: use Kaggle API if credentials are configured.
    3) --source academic-torrents: use aria2c to download the Academic Torrents torrent.

Run from project root:

    python scripts/download_yaleb_dataset.py --verify-only
    python scripts/download_yaleb_dataset.py --source kaggle
    python scripts/download_yaleb_dataset.py --source academic-torrents
    python scripts/download_yaleb_dataset.py --zip-path C:\\Users\\111\\Downloads\\CroppedYale.zip
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

SCRIPT_VERSION = "2026-07-07-yaleB-downloader-v2-kaggle-academic-torrents"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "data"
DEFAULT_TARGET = DEFAULT_DATA_ROOT / "yaleB"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "datasets" / "_downloads" / "yaleb"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".ppm", ".tif", ".tiff"}
SUBJECT_RE = re.compile(r"yaleB(\d+)", re.IGNORECASE)

ACADEMIC_TORRENTS_HASH = "aad8bf8e6ee5d8a3bf46c7ab5adfacdd8ad36247"
ACADEMIC_TORRENTS_TORRENT_URL = (
    f"https://academictorrents.com/download/{ACADEMIC_TORRENTS_HASH}.torrent"
)
DEFAULT_KAGGLE_SLUG = "jensdhondt/extendedyaleb-cropped-full"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _safe_extract_zip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        dest_resolved = dest.resolve()
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if not str(target).startswith(str(dest_resolved)):
                raise RuntimeError(f"Unsafe zip member: {member.filename}")
        zf.extractall(dest)


def _find_archives(root: Path) -> list[Path]:
    exts = {".zip"}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def _find_subject_dirs(root: Path) -> list[Path]:
    subject_dirs: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        if SUBJECT_RE.search(p.name):
            imgs = [q for q in p.iterdir() if q.is_file() and q.suffix.lower() in IMAGE_EXTENSIONS]
            if imgs:
                subject_dirs.append(p)
    return sorted(subject_dirs, key=lambda x: str(x).lower())


def _copy_subject_dirs(subject_dirs: list[Path], target: Path, force: bool = False) -> None:
    if target.exists() and force:
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    seen: dict[str, int] = {}
    for src in subject_dirs:
        m = SUBJECT_RE.search(src.name) or SUBJECT_RE.search(str(src))
        if not m:
            continue
        name = f"yaleB{int(m.group(1)):02d}"
        # Avoid collisions from nested duplicate layouts.
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name}_copy{seen[name]}"
        dst = target / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.txt", "*.mat"))


def _summarize(target: Path) -> tuple[int, int]:
    subject_dirs = _find_subject_dirs(target)
    n_subjects = len(subject_dirs)
    n_images = 0
    print("\nYale B layout summary")
    print(f"  target       : {target}")
    print(f"  subjects     : {n_subjects}")
    for d in subject_dirs[:5]:
        imgs = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        n_images += len(imgs)
    n_images = sum(
        len([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
        for d in subject_dirs
    )
    print(f"  images       : {n_images}")
    if subject_dirs:
        first = subject_dirs[0]
        last = subject_dirs[-1]
        first_n = len([p for p in first.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
        last_n = len([p for p in last.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
        print(f"  first subject: {first.relative_to(target)} ({first_n} images)")
        print(f"  last subject : {last.relative_to(target)} ({last_n} images)")
    return n_subjects, n_images


def install_from_tree(tree_root: Path, target: Path, force: bool = False) -> None:
    subject_dirs = _find_subject_dirs(tree_root)
    if not subject_dirs:
        raise RuntimeError(
            f"No yaleBXX subject folders containing images were found under: {tree_root}\n"
            "Expected something like CroppedYale/yaleB01/*.pgm or yaleB01/*.png."
        )
    _copy_subject_dirs(subject_dirs, target, force=force)
    n_subjects, n_images = _summarize(target)
    if n_subjects < 2 or n_images == 0:
        raise RuntimeError("Installed Yale B data look incomplete.")


def install_from_zip(zip_path: Path, target: Path, cache_dir: Path, force: bool = False) -> None:
    zip_path = zip_path.resolve()
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    extract_root = cache_dir / "extract_zip"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    print(f"Extracting: {zip_path}")
    _safe_extract_zip(zip_path, extract_root)

    # Some Kaggle downloads contain another archive inside.
    inner_archives = _find_archives(extract_root)
    if inner_archives and not _find_subject_dirs(extract_root):
        for arch in inner_archives:
            try:
                inner_dest = extract_root / f"_inner_{arch.stem}"
                print(f"Extracting nested archive: {arch}")
                _safe_extract_zip(arch, inner_dest)
            except zipfile.BadZipFile:
                pass

    install_from_tree(extract_root, target, force=force)


def download_url(url: str, out_path: Path, timeout: int = 90) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {url}")
    with urllib.request.urlopen(url, timeout=timeout) as response:
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
                    print(
                        f"  {downloaded / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB",
                        end="\r",
                    )
    print(f"\nSaved: {out_path}")


def install_from_kaggle(slug: str, target: Path, cache_dir: Path, force: bool = False) -> None:
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise RuntimeError(
            "Kaggle CLI not found. Install with:\n"
            "  python -m pip install kaggle\n"
            "Then configure %USERPROFILE%\\.kaggle\\kaggle.json."
        )
    extract_root = cache_dir / "kaggle"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    _run([kaggle, "datasets", "download", "-d", slug, "-p", str(extract_root), "--unzip"])

    # If Kaggle only downloaded zip instead of unzipping, extract it.
    archives = _find_archives(extract_root)
    for arch in archives:
        try:
            inner = extract_root / f"_extract_{arch.stem}"
            print(f"Extracting Kaggle archive: {arch}")
            _safe_extract_zip(arch, inner)
        except zipfile.BadZipFile:
            pass

    install_from_tree(extract_root, target, force=force)


def install_from_academic_torrents(target: Path, cache_dir: Path, force: bool = False) -> None:
    aria2c = shutil.which("aria2c")
    if aria2c is None:
        torrent_path = cache_dir / "CroppedYale.academic.torrent"
        try:
            download_url(ACADEMIC_TORRENTS_TORRENT_URL, torrent_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Could not download torrent file automatically: {exc}")
        raise RuntimeError(
            "aria2c was not found. Install aria2, then rerun this command.\n\n"
            "Recommended Windows options:\n"
            "  winget install aria2.aria2\n"
            "or:\n"
            "  conda install -c conda-forge aria2\n\n"
            "Then rerun:\n"
            "  python .\\scripts\\download_yaleb_dataset.py --source academic-torrents --force\n\n"
            f"Torrent URL: {ACADEMIC_TORRENTS_TORRENT_URL}"
        )

    torrent_path = cache_dir / "CroppedYale.academic.torrent"
    if not torrent_path.exists() or force:
        download_url(ACADEMIC_TORRENTS_TORRENT_URL, torrent_path)

    download_dir = cache_dir / "academic_torrents"
    download_dir.mkdir(parents=True, exist_ok=True)
    _run([
        aria2c,
        "--seed-time=0",
        "--max-connection-per-server=4",
        "--dir", str(download_dir),
        str(torrent_path),
    ])

    archives = _find_archives(download_dir)
    if archives:
        # Prefer CroppedYale.zip if present.
        archives = sorted(archives, key=lambda p: ("croppedyale" not in p.name.lower(), p.name.lower()))
        install_from_zip(archives[0], target, cache_dir, force=force)
        return

    install_from_tree(download_dir, target, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install Cropped Extended Yale B into datasets/data/yaleB.")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET, help="Target directory. Default: datasets/data/yaleB")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Download/extraction cache directory.")
    parser.add_argument("--source", choices=["manual", "kaggle", "academic-torrents"], default="manual")
    parser.add_argument("--zip-path", type=Path, default=None, help="Install from an already downloaded zip file.")
    parser.add_argument("--tree-path", type=Path, default=None, help="Install from an already extracted folder containing yaleBXX folders.")
    parser.add_argument("--url", default=None, help="Direct zip URL to download and install from.")
    parser.add_argument("--kaggle-slug", default=DEFAULT_KAGGLE_SLUG, help="Kaggle dataset slug.")
    parser.add_argument("--force", action="store_true", help="Overwrite target/cache if needed.")
    parser.add_argument("--verify-only", action="store_true", help="Only summarize the existing target directory.")
    args = parser.parse_args()

    print(f"download_yaleb_dataset.py version: {SCRIPT_VERSION}")
    target = args.target.resolve()
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        _summarize(target)
        return

    if target.exists() and list(target.rglob("*")) and not args.force:
        n_subjects, n_images = _summarize(target)
        if n_subjects >= 2 and n_images > 0:
            print("Existing Yale B data found. Use --force to reinstall.")
            return

    if args.tree_path is not None:
        install_from_tree(args.tree_path.resolve(), target, force=args.force)
    elif args.zip_path is not None:
        install_from_zip(args.zip_path.resolve(), target, cache_dir, force=args.force)
    elif args.url is not None:
        zip_path = cache_dir / Path(args.url).name
        download_url(args.url, zip_path)
        install_from_zip(zip_path, target, cache_dir, force=args.force)
    elif args.source == "kaggle":
        install_from_kaggle(args.kaggle_slug, target, cache_dir, force=args.force)
    elif args.source == "academic-torrents":
        install_from_academic_torrents(target, cache_dir, force=args.force)
    else:
        raise RuntimeError(
            "No local zip/folder was provided. Choose one of:\n"
            "  python scripts/download_yaleb_dataset.py --source kaggle\n"
            "  python scripts/download_yaleb_dataset.py --source academic-torrents\n"
            "  python scripts/download_yaleb_dataset.py --zip-path C:\\path\\to\\CroppedYale.zip\n"
            "  python scripts/download_yaleb_dataset.py --tree-path C:\\path\\to\\CroppedYale"
        )

    print("\nDone. Use this data root in experiments:")
    print("  --dataset yaleB --data-root datasets/data/yaleB")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
