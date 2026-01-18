#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_row_f32(path: Path, i: int, d: int) -> np.ndarray:
    with open(path, "rb") as f:
        f.seek(i * d * 4)
        buf = f.read(d * 4)
    if len(buf) != d * 4:
        raise ValueError(f"Row {i} out of range (file too small?)")
    return np.frombuffer(buf, dtype="<f4")


def read_label_u8(path: Path, i: int) -> int:
    with open(path, "rb") as f:
        f.seek(i)
        b = f.read(1)
    if len(b) != 1:
        raise ValueError(f"Label index {i} out of range")
    return b[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("data"))
    ap.add_argument("--index", type=int, default=0, help="sample index i")
    ap.add_argument("--out", type=Path, default=Path("sample.png"))
    ap.add_argument("--shape", type=str, default="64,64", help="H,W (default 64,64)")
    args = ap.parse_args()

    meta = json.loads((args.dir / "meta.json").read_text(encoding="utf-8"))
    classes = meta.get("classes", [])
    d = int(meta["d"])

    h, w = (int(x) for x in args.shape.split(","))
    if h * w != d:
        raise ValueError(f"shape {h}x{w} != d={d}")

    x_path = args.dir / meta["X"]["file"]
    y_path = args.dir / meta["y"]["file"]

    row = read_row_f32(x_path, args.index, d)
    label = read_label_u8(y_path, args.index)
    label_name = classes[label] if classes and label < len(classes) else str(label)

    print(f"index: {args.index}")
    print(f"label: {label} ({label_name})")
    print(f"min/max: {float(row.min()):.4f} / {float(row.max()):.4f}")

    img_u8 = (row.reshape(h, w) * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_u8, mode="L").save(args.out)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()


