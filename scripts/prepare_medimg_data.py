#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def iter_image_files(root: Path) -> list[tuple[str, Path]]:
    items = []
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        class_name = class_dir.name
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in ".jpeg":
                items.append((class_name, p))
    return items


def load_64x64_grayscale_vector(img_path: Path) -> np.ndarray:
    with Image.open(img_path) as im:
        im = im.convert("L")  # grayscale
        if im.size != (64, 64):
            raise ValueError(f"Expected 64x64, got {im.size} for {img_path}")
        arr = np.asarray(im, dtype=np.float32) / 255.0  # [0,1]
        return arr.reshape(-1)  # (4096,)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/images"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/medimg_bin"))
    args = ap.parse_args()

    items = iter_image_files(args.root)
    if not items:
        raise SystemExit(f"No images found under {args.root}")

    classes = sorted({c for c, _ in items})
    class_to_idx = {c: i for i, c in enumerate(classes)}

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    paths: list[str] = []

    for class_name, img_path in items:
        vec = load_64x64_grayscale_vector(img_path)
        X_list.append(vec)
        y_list.append(class_to_idx[class_name])
        paths.append(str(img_path))

    X = np.ascontiguousarray(np.stack(X_list).astype("<f4"))  # (N,4096)
    y = np.ascontiguousarray(np.array(y_list, dtype=np.uint8))     # (N,)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    x_path = out_dir / "X_f32.bin"
    y_path = out_dir / "y_u8.bin"
    meta_path = out_dir / "meta.json"

    # Raw binary writes
    X.tofile(x_path)
    y.tofile(y_path)

    meta = {
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),              # 4096
        "classes": classes,
        "class_to_idx": class_to_idx,
        "X": {"file": x_path.name, "dtype": "f32", "endian": "little", "layout": "row-major"},
        "y": {"file": y_path.name, "dtype": "u8"},
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote: {x_path} ({x_path.stat().st_size} bytes)")
    print(f"Wrote: {y_path} ({y_path.stat().st_size} bytes)")
    print(f"Wrote: {meta_path}")
    print(f"X shape: {X.shape}  y shape: {y.shape}")


if __name__ == "__main__":
    main()
