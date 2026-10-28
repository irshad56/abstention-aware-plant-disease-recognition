"""Safe archive extractor protecting against zip-slip and path traversal."""

import argparse
import os
from pathlib import Path
import zipfile


def safe_extract(zip_path: Path, target_dir: Path):
    """Safely extracts zip archive without allowing path traversal vulnerabilities."""
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():
            # Validate target path
            extracted_path = (target_dir / member.filename).resolve()
            if not str(extracted_path).startswith(str(target_dir)):
                raise ValueError(f"Security error: Attempted Path Traversal in archive: {member.filename}")
            z.extract(member, target_dir)
            print(f"Extracted: {member.filename}")


def main():
    parser = argparse.ArgumentParser(description="Safely extract model archives (e.g., CLIP artifacts).")
    parser.add_argument("archive", type=str, help="Path to .zip archive to extract")
    parser.add_argument("--dest", "-d", type=str, default="artifacts/models/clip_vit_b32", help="Destination folder")
    args = parser.parse_args()

    arch_p = Path(args.archive)
    dest_p = Path(args.dest)

    if not arch_p.exists():
        print(f"Error: Archive not found: {arch_p}")
        return

    print(f"Extracting {arch_p} safely to {dest_p}...")
    safe_extract(arch_p, dest_p)
    print("Extraction completed successfully.")


if __name__ == "__main__":
    main()
