"""Compress a grayscale image with a truncated SVD computed by this toolkit.

    python examples/image_compression.py                 # scikit-learn's bundled sample photo
    python examples/image_compression.py --image me.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from linalg import svd


def load_gray(path):
    if path:
        img = plt.imread(path).astype(float)
    else:
        from sklearn.datasets import load_sample_image
        img = load_sample_image("china.jpg").astype(float)[::2, ::2]  # downsample to keep pure-Python SVD fast
    if img.ndim == 3:
        img = img[..., :3] @ np.array([0.299, 0.587, 0.114])
    return img / img.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--ranks", type=int, nargs="+", default=[5, 20, 50])
    ap.add_argument("--out", default="examples/svd_compression.png")
    args = ap.parse_args()

    img = load_gray(args.image)
    m, n = img.shape
    U, s, Vt = svd(img)
    fig, axes = plt.subplots(1, len(args.ranks) + 1, figsize=(4 * (len(args.ranks) + 1), 4))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Original ({m}x{n})")
    for ax, k in zip(axes[1:], args.ranks):
        approx = (U[:, :k] * s[:k]) @ Vt[:k]
        err = np.linalg.norm(img - approx) / np.linalg.norm(img)
        stored = k * (m + n + 1) / (m * n)
        ax.imshow(np.clip(approx, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Rank {k}: {stored:.1%} of storage")
        print(f"rank {k:>3}: relative error {err:.3f}, storage {stored:.1%} of original")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
