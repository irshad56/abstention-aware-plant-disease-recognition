"""Upload authoritative_plant_disease_models.zip to GitHub Release v2.0.0."""

import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

TOKEN = os.environ.get("GITHUB_TOKEN", "")
REPO = "irshad56/abstention-aware-plant-disease-recognition"
TAG = "v2.0.0"
ZIP_PATH = Path("authoritative_plant_disease_models.zip")

HEADERS_BASE = {
    "Authorization": f"token {TOKEN}",
    "User-Agent": "PlantDiseaseUploader/2.0",
    "Accept": "application/vnd.github.v3+json",
}


def get_or_create_release():
    # Check if release already exists
    list_url = f"https://api.github.com/repos/{REPO}/releases"
    req = urllib.request.Request(list_url, headers=HEADERS_BASE)
    try:
        with urllib.request.urlopen(req) as resp:
            releases = json.load(resp)
            for r in releases:
                if r["tag_name"] == TAG:
                    print(f"Found existing release {TAG} (id={r['id']})")
                    return r
    except Exception as e:
        print(f"Warning searching releases: {e}")

    # Create new release
    print(f"Creating release {TAG}...")
    create_url = f"https://api.github.com/repos/{REPO}/releases"
    payload = json.dumps({
        "tag_name": TAG,
        "name": f"{TAG} - Authoritative 4-Level Deep Ensemble & Decision Layer",
        "body": "Authoritative release containing all 5 deep disease vision models (ConvNeXtV2-Tiny, DINOv2-Small, EfficientNetV2-S, MaxViT-Tiny, Swin-Tiny), ResNet18 domain-adapted leaf verifier, calibrated 290-dim decision gates, and literature evidence reports.",
        "draft": False,
        "prerelease": False,
    }).encode("utf-8")

    req = urllib.request.Request(create_url, data=payload, headers={**HEADERS_BASE, "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        rel = json.load(resp)
        print(f"Created release {TAG} (id={rel['id']})")
        return rel


def upload_asset(release):
    upload_url_template = release["upload_url"]
    upload_url = upload_url_template.split("{")[0] + f"?name={ZIP_PATH.name}"

    # Check if asset already exists in release
    for asset in release.get("assets", []):
        if asset["name"] == ZIP_PATH.name:
            print(f"Asset {ZIP_PATH.name} already exists (id={asset['id']}), deleting prior asset to re-upload...")
            del_req = urllib.request.Request(asset["url"], headers=HEADERS_BASE, method="DELETE")
            with urllib.request.urlopen(del_req) as del_resp:
                print(f"Deleted previous asset: {del_resp.status}")

    file_size = ZIP_PATH.stat().st_size
    print(f"Uploading {ZIP_PATH.name} ({file_size / (1024*1024):.2f} MB) to {upload_url}...")

    # Upload via binary stream
    with open(ZIP_PATH, "rb") as f:
        data = f.read()

    req = urllib.request.Request(
        upload_url,
        data=data,
        headers={
            **HEADERS_BASE,
            "Content-Type": "application/zip",
            "Content-Length": str(len(data)),
        },
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.load(resp)
        print("Upload successful!")
        print("Download URL:", result.get("browser_download_url"))
        return result.get("browser_download_url")


if __name__ == "__main__":
    rel = get_or_create_release()
    url = upload_asset(rel)
    print("\nAUTHORITATIVE_MODEL_URL:", url)
