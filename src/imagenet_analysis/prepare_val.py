"""Verify the official validation archive and extract its 50,000 images."""
import hashlib
import json
import re
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "ILSVRC2012_img_val.tar"
PART = ROOT / "ILSVRC2012_img_val.tar.part"
EXPECTED_SIZE = 6744924160
EXPECTED_MD5 = "29b22e2961454d5413ddabcf34fc5622"
SOURCE = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"


def main():
    source = ARCHIVE if ARCHIVE.exists() else PART
    if source.stat().st_size != EXPECTED_SIZE:
        raise ValueError(f"Incomplete archive: {source.stat().st_size} bytes")
    print("Checking archive MD5...", flush=True)
    with source.open("rb") as stream:
        actual_md5 = hashlib.file_digest(stream, "md5").hexdigest()
    if actual_md5 != EXPECTED_MD5:
        raise ValueError(f"Archive MD5 mismatch: {actual_md5}")
    if source == PART:
        source.rename(ARCHIVE)
    val = ROOT / "val"
    val.mkdir(exist_ok=True)
    expected_names = {f"ILSVRC2012_val_{i:08d}.JPEG" for i in range(1, 50001)}
    total_bytes = 0
    with tarfile.open(ARCHIVE, "r:") as tar:
        members = tar.getmembers()
        if len(members) != 50000 or {m.name for m in members} != expected_names:
            raise ValueError("Unexpected archive filenames or image count")
        if any(not m.isfile() or not re.fullmatch(r"ILSVRC2012_val_\d{8}\.JPEG", m.name) for m in members):
            raise ValueError("Archive contains an unexpected member")
        for i, member in enumerate(members, 1):
            destination = val / member.name
            with tar.extractfile(member) as stream:
                if destination.exists():
                    with destination.open("rb") as existing:
                        if hashlib.file_digest(existing, "sha256").digest() != hashlib.file_digest(stream, "sha256").digest():
                            raise ValueError(f"Existing image differs from archive: {destination.name}")
                else:
                    with destination.open("xb") as output:
                        shutil.copyfileobj(stream, output, length=1024 * 1024)
            if destination.stat().st_size != member.size:
                raise ValueError(f"Extracted image size mismatch: {destination.name}")
            total_bytes += member.size
            if i % 10000 == 0:
                print(f"Extracted and checked {i:,}/50,000 images", flush=True)
    if {p.name for p in val.iterdir()} != expected_names:
        raise ValueError("Unexpected files in validation directory")
    manifest = {
        "dataset": "ILSVRC2012 validation images",
        "source_url": SOURCE,
        "archive": ARCHIVE.name,
        "archive_bytes": EXPECTED_SIZE,
        "md5": actual_md5,
        "checksum_reference": "https://github.com/pytorch/vision/blob/main/torchvision/datasets/imagenet.py",
        "images": 50000,
        "extracted_image_bytes": total_bytes,
        "image_directory": "val",
        "bbox_directory": "bbox",
        "bbox_status": "User will supply annotations separately",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (ROOT / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("COMPLETE: official archive verified; 50,000 validation images ready.", flush=True)


if __name__ == "__main__":
    main()
