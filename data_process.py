import argparse
import os
import re
import sys

import cv2


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v"}


def _natural_key(text):
    parts = re.split(r"(\d+)", text)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def list_videos(dir_path):
    if not os.path.isdir(dir_path):
        return []
    files = []
    for name in os.listdir(dir_path):
        _, ext = os.path.splitext(name)
        if ext.lower() in VIDEO_EXTS:
            files.append(os.path.join(dir_path, name))
    files.sort(key=lambda p: _natural_key(os.path.basename(p)))
    return files


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def extract_frames(video_path, out_dir, start_index=1, pad=6):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"warn: cannot open {video_path}", file=sys.stderr)
        return 0

    ensure_dir(out_dir)
    index = start_index
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        filename = f"frame_{index:0{pad}d}.jpg"
        out_path = os.path.join(out_dir, filename)
        if cv2.imwrite(out_path, frame):
            saved += 1
        index += 1
    cap.release()
    return saved


def process_videos(root_dir, out_root):
    categories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    categories.sort(key=_natural_key)
    total = 0
    for category in categories:
        category_dir = os.path.join(root_dir, category)
        videos = list_videos(category_dir)
        for video_path in videos:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            out_dir = os.path.join(out_root, category, video_name)
            count = extract_frames(video_path, out_dir)
            print(f"{video_path} -> {out_dir} ({count} frames)")
            total += count
    print(f"total frames: {total}")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos into images directory.")
    parser.add_argument(
        "--videos-dir",
        default="videos",
        help="Root directory containing video subfolders (default: videos).",
    )
    parser.add_argument(
        "--out-dir",
        default="images",
        help="Output root directory for frames (default: images).",
    )
    args = parser.parse_args()
    process_videos(args.videos_dir, args.out_dir)


if __name__ == "__main__":
    main()
