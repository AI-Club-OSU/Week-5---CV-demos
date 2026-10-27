"""Download the three MediaPipe model files into ./models.

Run once, before the workshop:  python download_models.py
The URLs are the official ones from Google's MediaPipe docs.
"""

from __future__ import annotations

import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

MODELS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    ),
    "gesture_recognizer.task": (
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
        "gesture_recognizer/float16/latest/gesture_recognizer.task"
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    ),
}

MODELS_DIR = Path(__file__).resolve().parent / "models"


def open_url(url: str):
    """Open a URL with the system's certificates. If those can't verify the site
    (common with python.org installs on a Mac), retry with certifi's."""
    try:
        return urllib.request.urlopen(url, timeout=60)
    except urllib.error.URLError as error:
        if not isinstance(error.reason, ssl.SSLCertVerificationError):
            raise
        try:
            import certifi
        except ImportError:
            raise error from None
        context = ssl.create_default_context(cafile=certifi.where())
        return urllib.request.urlopen(url, context=context, timeout=60)


def download(url: str, dest: Path) -> None:
    """Download to a .part file first, so a failed download never looks finished."""
    part = dest.with_name(dest.name + ".part")
    with open_url(url) as response:
        with open(part, "wb") as out:
            while True:
                chunk = response.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
    part.replace(dest)


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    failed = []
    for name, url in MODELS.items():
        dest = MODELS_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"ok    {name} (already downloaded)")
            continue
        print(f"get   {name} ...", end=" ", flush=True)
        try:
            download(url, dest)
            print(f"{dest.stat().st_size / 1e6:.1f} MB")
        except Exception as error:  # noqa: BLE001 - show any network error plainly
            print(f"failed ({error})")
            failed.append((name, url))

    if failed:
        print(f"\nSome downloads failed. Open these links in a browser and save the files in {MODELS_DIR}:")
        for name, url in failed:
            print(f"  {name}\n    {url}")
        sys.exit(1)
    print(f"\nAll models are in {MODELS_DIR}")


if __name__ == "__main__":
    main()
