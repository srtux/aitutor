#!/usr/bin/env python3
"""Download and vendor local offline model assets and web runtimes for AiTutor.

This script ensures all models and runtimes required for 100% offline edge
operation are downloaded and verified:
1. MediaPipe FaceLandmarker model (.task)
2. MediaPipe Tasks-Vision JS + WASM bundle (offline browser execution)
3. Kokoro ONNX v1.0 TTS model and voice binaries
"""

import argparse
import io
from pathlib import Path
import shutil
import sys
import tarfile
import urllib.error
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent

# Asset definitions
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_PATH = REPO_ROOT / "frontend" / "assets" / "face_landmarker.task"

MEDIAPIPE_VISION_TAR_URL = (
    "https://registry.npmjs.org/@mediapipe/tasks-vision/-/"
    "tasks-vision-0.10.18.tgz"
)
MEDIAPIPE_VENDOR_DIR = REPO_ROOT / "frontend" / "vendor" / "mediapipe"

KOKORO_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/kokoro-v1.0.onnx"
)
KOKORO_MODEL_PATH = REPO_ROOT / "models" / "kokoro-v1.0.onnx"

KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/voices-v1.0.bin"
)
KOKORO_VOICES_PATH = REPO_ROOT / "models" / "voices-v1.0.bin"


def download_file(url: str, dest_path: Path, timeout: int = 120) -> bool:
    """Download a file with streaming progress and write to destination."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    print(f"Downloading {dest_path.name} from {url}...")
    headers = {"User-Agent": "AiTutor-Asset-Downloader/1.0"}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            total_size = response.headers.get("Content-Length")
            total_bytes = int(total_size) if total_size else 0
            downloaded = 0
            block_size = 64 * 1024

            with open(temp_path, "wb") as f_out:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes > 0:
                        pct = (downloaded / total_bytes) * 100
                        mb_down = downloaded / (1024 * 1024)
                        mb_total = total_bytes / (1024 * 1024)
                        sys.stdout.write(
                            f"\r  -> {mb_down:.1f}MB / {mb_total:.1f}MB ({pct:.1f}%)"
                        )
                        sys.stdout.flush()

        if total_bytes > 0:
            sys.stdout.write("\n")
        temp_path.replace(dest_path)
        print(f"Successfully saved {dest_path.name} ({dest_path.stat().st_size:,} bytes).")
        return True
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"\nFailed to download {url}: {exc}", file=sys.stderr)
        if temp_path.exists():
            temp_path.unlink()
        return False


def setup_mediapipe_vision() -> bool:
    """Download and vendor the MediaPipe tasks-vision npm package."""
    bundle_mjs = MEDIAPIPE_VENDOR_DIR / "vision_bundle.mjs"
    wasm_internal = MEDIAPIPE_VENDOR_DIR / "wasm" / "vision_wasm_internal.wasm"

    if bundle_mjs.is_file() and wasm_internal.is_file():
        if bundle_mjs.stat().st_size > 0 and wasm_internal.stat().st_size > 0:
            print(f"MediaPipe vision vendor bundle already present at {MEDIAPIPE_VENDOR_DIR}")
            return True

    print(f"Fetching MediaPipe tasks-vision package from {MEDIAPIPE_VISION_TAR_URL}...")
    headers = {"User-Agent": "AiTutor-Asset-Downloader/1.0"}
    req = urllib.request.Request(MEDIAPIPE_VISION_TAR_URL, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            tar_data = io.BytesIO(response.read())

        MEDIAPIPE_VENDOR_DIR.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=tar_data, mode="r:gz") as tar:
            for member in tar.getmembers():
                # Stripping leading "package/" directory prefix from tarball
                parts = Path(member.name).parts
                if not parts or parts[0] != "package" or len(parts) == 1:
                    continue
                rel_parts = parts[1:]
                target_file = MEDIAPIPE_VENDOR_DIR.joinpath(*rel_parts)

                if member.isdir():
                    target_file.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    f_in = tar.extractfile(member)
                    if f_in:
                        with open(target_file, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)

        print(f"Successfully extracted MediaPipe vision bundle into {MEDIAPIPE_VENDOR_DIR}")
        return True
    except (urllib.error.URLError, TimeoutError, OSError, tarfile.TarError) as exc:
        print(f"Failed to fetch MediaPipe vision package: {exc}", file=sys.stderr)
        return False


def verify_assets(skip_kokoro: bool = False) -> bool:
    """Verify all required assets exist and are non-empty."""
    required = [
        ("FaceLandmarker Task", FACE_LANDMARKER_PATH, 1024 * 1024),
        ("Vision Bundle MJS", MEDIAPIPE_VENDOR_DIR / "vision_bundle.mjs", 100 * 1024),
        ("Vision WASM Internal", MEDIAPIPE_VENDOR_DIR / "wasm" / "vision_wasm_internal.wasm", 100 * 1024),
    ]
    if not skip_kokoro:
        required.extend([
            ("Kokoro ONNX Model", KOKORO_MODEL_PATH, 10 * 1024 * 1024),
            ("Kokoro Voices Bin", KOKORO_VOICES_PATH, 10 * 1024 * 1024),
        ])

    all_ok = True
    print("\nVerifying model and vendor assets:")
    for name, path, min_size in required:
        if path.is_file() and path.stat().st_size >= min_size:
            print(f"  [OK] {name}: {path.name} ({path.stat().st_size:,} bytes)")
        else:
            size_str = f"{path.stat().st_size:,} bytes" if path.is_file() else "MISSING"
            print(f"  [FAIL] {name}: {path} ({size_str}, expected >= {min_size:,} bytes)")
            all_ok = False
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Download AiTutor offline models and vendor files.")
    parser.add_argument(
        "--skip-kokoro",
        action="store_true",
        help="Skip downloading Kokoro TTS model and voices (useful for minimal CI runs).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing assets without attempting download.",
    )
    args = parser.parse_args()

    if args.verify_only:
        return 0 if verify_assets(skip_kokoro=args.skip_kokoro) else 1

    # 1. MediaPipe FaceLandmarker model
    if not FACE_LANDMARKER_PATH.is_file() or FACE_LANDMARKER_PATH.stat().st_size == 0:
        if not download_file(FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH):
            print("Error downloading FaceLandmarker model.", file=sys.stderr)
            return 1
    else:
        print(f"FaceLandmarker model already present at {FACE_LANDMARKER_PATH}")

    # 2. MediaPipe Tasks-Vision runtime
    if not setup_mediapipe_vision():
        print("Error setting up MediaPipe tasks-vision runtime.", file=sys.stderr)
        return 1

    # 3. Kokoro ONNX model and voices
    if not args.skip_kokoro:
        if not KOKORO_MODEL_PATH.is_file() or KOKORO_MODEL_PATH.stat().st_size == 0:
            if not download_file(KOKORO_MODEL_URL, KOKORO_MODEL_PATH):
                print("Error downloading Kokoro ONNX model.", file=sys.stderr)
                return 1
        else:
            print(f"Kokoro model already present at {KOKORO_MODEL_PATH}")

        if not KOKORO_VOICES_PATH.is_file() or KOKORO_VOICES_PATH.stat().st_size == 0:
            if not download_file(KOKORO_VOICES_URL, KOKORO_VOICES_PATH):
                print("Error downloading Kokoro voices.", file=sys.stderr)
                return 1
        else:
            print(f"Kokoro voices already present at {KOKORO_VOICES_PATH}")

    return 0 if verify_assets(skip_kokoro=args.skip_kokoro) else 1


if __name__ == "__main__":
    sys.exit(main())
