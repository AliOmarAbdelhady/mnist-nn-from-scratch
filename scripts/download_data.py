"""Download the canonical MNIST IDX files from the cvdfoundation/mnist mirror.

The original yann.lecun.com host is frequently down; the cvdfoundation GitHub
mirror is the de-facto reliable source. Files arrive gzipped and are
decompressed in place so ``nn_from_scratch.data.load_idx`` can read them.

Usage:
    python scripts/download_data.py [--data-dir data]
"""

from __future__ import annotations

import argparse
import hashlib
import gzip
import shutil
import sys
import urllib.request
from pathlib import Path

# Reliable mirrors (tried in order). The original yann.lecun.com host is
# frequently down; the PyTorch S3 mirror (ossci-datasets) is the de-facto
# reliable source today. Files arrive gzipped and are decompressed in place
# so ``nn_from_scratch.data.load_idx`` can read them.
_FILES = [
    ("train-images-idx3-ubyte.gz", [
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
        "https://github.com/cvdfoundation/mnist/raw/main/train-images-idx3-ubyte.gz",
    ]),
    ("train-labels-idx1-ubyte.gz", [
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
        "https://github.com/cvdfoundation/mnist/raw/main/train-labels-idx1-ubyte.gz",
    ]),
    ("t10k-images-idx3-ubyte.gz", [
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
        "https://github.com/cvdfoundation/mnist/raw/main/t10k-images-idx3-ubyte.gz",
    ]),
    ("t10k-labels-idx1-ubyte.gz", [
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
        "https://github.com/cvdfoundation/mnist/raw/main/t10k-labels-idx1-ubyte.gz",
    ]),
]

# Expected (decompressed) sizes in bytes — quick integrity check.
_EXPECTED_BYTES = {
    "train-images-idx3-ubyte": 47040016,   # 60000*28*28 + 16 header
    "train-labels-idx1-ubyte": 60008,      # 60000 + 8 header
    "t10k-images-idx3-ubyte":  7840016,    # 10000*28*28 + 16 header
    "t10k-labels-idx1-ubyte":  10008,      # 10000 + 8 header
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "mnist-nn/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def try_download(urls, dest: Path) -> None:
    """Try each mirror URL until one succeeds."""
    last_err = None
    for url in urls:
        try:
            download(url, dest)
            return
        except Exception as e:  # noqa: BLE001
            print(f"    ({url}) failed: {e}")
            last_err = e
            if dest.exists():
                dest.unlink()
    raise RuntimeError(f"All mirrors failed for {dest.name}: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for gz_name, urls in _FILES:
        raw_name = gz_name[:-3]                       # strip .gz
        raw_path = data_dir / raw_name
        if raw_path.exists() and raw_path.stat().st_size == _EXPECTED_BYTES[raw_name]:
            print(f"  OK  {raw_name} already present ({raw_path.stat().st_size} B)")
            continue
        gz_path = data_dir / gz_name
        if not gz_path.exists():
            try_download(urls, gz_path)
        print(f"  decompressing {gz_name} -> {raw_name}")
        with gzip.open(gz_path, "rb") as src, open(raw_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        size = raw_path.stat().st_size
        ok = size == _EXPECTED_BYTES[raw_name]
        print(f"  {'OK ' if ok else 'WARN size mismatch'} {raw_name}: {size} B "
              f"(expected {_EXPECTED_BYTES[raw_name]})")
        # Remove the .gz to save space once verified.
        if ok and gz_path.exists():
            gz_path.unlink()

    print("\nDone. MNIST IDX files are in", data_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
