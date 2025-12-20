import argparse
import csv
import os
import sys
import shutil

import cv2
import numpy as np


def read_group_labels(labels_file):
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"labels file not found: {labels_file}")
    with open(labels_file, "r", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return []
    start = 0
    header = [cell.strip().lower() for cell in rows[0]]
    if header and header[0] in {"group_id", "path_1", "path"}:
        start = 1
        if header[0] == "path" and "group_id" not in header and "path_1" not in header:
            raise ValueError("labels file looks like per-image format; expected group labels.")
    entries = []
    for row in rows[start:]:
        if len(row) < 2:
            continue
        group_id = row[0]
        label = row[1]
        paths = []
        if len(row) >= 7:
            paths = [cell for cell in row[2:7] if cell]
        entries.append({"group_id": group_id, "label": label, "paths": paths})
    return entries


def label_to_int(value):
    text = str(value).strip().lower()
    if text in {"normal", "0"}:
        return 0
    if text in {"abnormal", "1"}:
        return 1
    raise ValueError(f"unknown label: {value}")


def all_files_exist(paths, images_dir):
    for rel_path in paths:
        abs_path = os.path.join(images_dir, rel_path)
        if not os.path.isfile(abs_path):
            return False
    return True


def load_images(paths, resize=False, pad=False, base_size=None, gray=False):
    images = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            return None, base_size
        if gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(img)

    if base_size is None:
        base_size = images[0].shape[:2]

    height, width = base_size
    resized = []
    for img in images:
        if img.shape[:2] != (height, width):
            if pad:
                if img.shape[0] > height or img.shape[1] > width:
                    return None, base_size
                pad_bottom = height - img.shape[0]
                pad_right = width - img.shape[1]
                img = cv2.copyMakeBorder(
                    img,
                    0,
                    pad_bottom,
                    0,
                    pad_right,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )
            elif resize:
                img = cv2.resize(img, (width, height))
            else:
                return None, base_size
        if gray and img.ndim == 2:
            img = img[:, :, None]
        resized.append(img)
    return resized, base_size


def build_features(images, dtype):
    diffs = []
    if dtype == np.uint8:
        prev = images[0].astype(np.int16)
        for img in images[1:]:
            current = img.astype(np.int16)
            diff = current - prev
            diff = np.clip((diff + 255) / 2.0, 0, 255).astype(np.uint8)
            diffs.append(diff)
            prev = current
        return np.stack(diffs, axis=0)

    if dtype == np.int16:
        prev = images[0].astype(np.int16)
        for img in images[1:]:
            current = img.astype(np.int16)
            diffs.append(current - prev)
            prev = current
        return np.stack(diffs, axis=0)

    prev = images[0].astype(np.float32)
    for img in images[1:]:
        current = img.astype(np.float32)
        diffs.append(current - prev)
        prev = current
    stacked = np.stack(diffs, axis=0)
    return stacked.astype(dtype, copy=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Build diff-image features and labels as .npy.")
    parser.add_argument(
        "--images-dir",
        default="images",
        help="Root directory containing image folders (default: images).",
    )
    parser.add_argument(
        "--labels-file",
        default="labels.csv",
        help="Group labels file produced by label.py (default: labels.csv).",
    )
    parser.add_argument(
        "--out-features",
        default="features.npy",
        help="Output .npy file for features (default: features.npy).",
    )
    parser.add_argument(
        "--out-labels",
        default="labels.npy",
        help="Output .npy file for labels (default: labels.npy).",
    )
    parser.add_argument(
        "--resize",
        action="store_true",
        help="Resize images to the first image size if shapes differ.",
    )
    parser.add_argument(
        "--pad",
        action="store_true",
        help="Pad images to target size instead of resizing.",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        help="Force resize to fixed size (height width).",
    )
    parser.add_argument(
        "--gray",
        action="store_true",
        help="Convert images to grayscale before differencing.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "int16", "uint8"),
        default="float32",
        help="Output dtype for diff features (default: float32).",
    )
    return parser.parse_args()


def compute_max_size(entries, images_dir):
    max_h = 0
    max_w = 0
    seen = set()
    for entry in entries:
        for rel_path in entry["paths"]:
            if not rel_path or rel_path in seen:
                continue
            seen.add(rel_path)
            abs_path = os.path.join(images_dir, rel_path)
            if not os.path.isfile(abs_path):
                continue
            img = cv2.imread(abs_path)
            if img is None:
                continue
            height, width = img.shape[:2]
            if height > max_h:
                max_h = height
            if width > max_w:
                max_w = width
    if max_h == 0 or max_w == 0:
        return None
    return (max_h, max_w)


def main():
    args = parse_args()
    images_dir = os.path.abspath(args.images_dir)
    entries = read_group_labels(args.labels_file)
    if not entries:
        print("error: no group labels found.", file=sys.stderr)
        return 1

    if args.size:
        base_size = (args.size[0], args.size[1])
    elif args.pad:
        base_size = compute_max_size(entries, images_dir)
        if base_size:
            print(f"pad target size: {base_size[0]}x{base_size[1]}", file=sys.stderr)
    else:
        base_size = None
    channels = None
    resize_needed = (args.resize or args.size is not None) and not args.pad
    output_dtype = np.dtype(args.dtype)
    if output_dtype == np.uint8:
        print("note: uint8 diff uses scaled mapping (diff + 255) / 2.", file=sys.stderr)
    gray = args.gray
    for entry in entries:
        paths = entry["paths"]
        if len(paths) != 5:
            continue
        if not all_files_exist(paths, images_dir):
            continue
        abs_paths = [os.path.join(images_dir, rel_path) for rel_path in paths]
        images, base_size = load_images(
            abs_paths,
            resize=resize_needed,
            pad=args.pad,
            base_size=base_size,
            gray=gray,
        )
        if images is None:
            continue
        channels = images[0].shape[2] if images[0].ndim == 3 else 1
        break

    if base_size is None or channels is None:
        print("error: no valid feature-label pairs created.", file=sys.stderr)
        return 1

    expected_count = 0
    for entry in entries:
        paths = entry["paths"]
        if len(paths) != 5:
            continue
        try:
            label_to_int(entry["label"])
        except ValueError:
            continue
        if not all_files_exist(paths, images_dir):
            continue
        expected_count += 1

    if expected_count == 0:
        print("error: no valid feature-label pairs created.", file=sys.stderr)
        return 1

    feature_shape = (expected_count, 4, base_size[0], base_size[1], channels)
    expected_bytes = (
        expected_count
        * 4
        * base_size[0]
        * base_size[1]
        * channels
        * output_dtype.itemsize
    )
    expected_bytes += expected_count * np.dtype(np.int64).itemsize
    out_dir = os.path.dirname(os.path.abspath(args.out_features)) or "."
    free_bytes = shutil.disk_usage(out_dir).free
    if expected_bytes > free_bytes:
        needed_gb = expected_bytes / 1024**3
        free_gb = free_bytes / 1024**3
        print(
            f"error: not enough disk space. need ~{needed_gb:.1f} GB, "
            f"free ~{free_gb:.1f} GB. "
            "Use --size/--gray/--dtype or choose another output path.",
            file=sys.stderr,
        )
        return 1
    features_arr = np.lib.format.open_memmap(
        args.out_features,
        mode="w+",
        dtype=output_dtype,
        shape=feature_shape,
    )
    labels_arr = np.lib.format.open_memmap(
        args.out_labels,
        mode="w+",
        dtype=np.int64,
        shape=(expected_count,),
    )

    write_idx = 0
    skipped = 0
    for entry in entries:
        paths = entry["paths"]
        if len(paths) != 5:
            skipped += 1
            continue
        abs_paths = [os.path.join(images_dir, rel_path) for rel_path in paths]
        images, _ = load_images(
            abs_paths,
            resize=resize_needed,
            pad=args.pad,
            base_size=base_size,
            gray=gray,
        )
        if images is None:
            skipped += 1
            continue
        try:
            label_value = label_to_int(entry["label"])
        except ValueError:
            skipped += 1
            continue
        features_arr[write_idx] = build_features(images, output_dtype)
        labels_arr[write_idx] = label_value
        write_idx += 1
        if write_idx % 200 == 0:
            features_arr.flush()
            labels_arr.flush()

    features_arr.flush()
    labels_arr.flush()

    final_shape = (write_idx, 4, base_size[0], base_size[1], channels)
    if write_idx < expected_count:
        tmp_features = args.out_features + ".tmp"
        tmp_labels = args.out_labels + ".tmp"
        trimmed_features = np.lib.format.open_memmap(
            tmp_features,
            mode="w+",
            dtype=output_dtype,
            shape=final_shape,
        )
        trimmed_labels = np.lib.format.open_memmap(
            tmp_labels,
            mode="w+",
            dtype=np.int64,
            shape=(write_idx,),
        )
        chunk = 50
        for start in range(0, write_idx, chunk):
            end = min(write_idx, start + chunk)
            trimmed_features[start:end] = features_arr[start:end]
            trimmed_labels[start:end] = labels_arr[start:end]
        trimmed_features.flush()
        trimmed_labels.flush()
        os.replace(tmp_features, args.out_features)
        os.replace(tmp_labels, args.out_labels)
    print(
        f"saved features: {args.out_features} shape={final_shape}, "
        f"labels: {args.out_labels} shape=({write_idx},), "
        f"skipped={skipped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
