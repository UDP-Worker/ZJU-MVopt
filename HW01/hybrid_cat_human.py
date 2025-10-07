import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def _load_image(path: Path, max_size: int | None = 1024) -> np.ndarray:
    """Load an image in BGR format and optionally resize while keeping aspect ratio."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {path}")

    if max_size is not None:
        h, w = image.shape[:2]
        longest = max(h, w)
        if longest > max_size:
            scale = max_size / float(longest)
            new_size = (int(round(w * scale)), int(round(h * scale)))
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            image = cv2.resize(image, new_size, interpolation=interpolation)
    return image


def _cascade_path() -> Path:
    data_root = getattr(getattr(cv2, "data", object()), "haarcascades", None)
    if data_root is None:
        cv2_base = Path(cv2.__file__).resolve().parent
        data_root = cv2_base / "data"
    return Path(data_root) / "haarcascade_frontalface_default.xml"


def _detect_face_ellipse(image: np.ndarray) -> np.ndarray:
    """Detect the dominant face region and return a soft elliptical mask."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = None
    cascade_path = _cascade_path()
    if cascade_path.exists():
        cascade = cv2.CascadeClassifier(str(cascade_path))

    faces: Tuple[int, int, int, int] | Tuple = ()
    if cascade is not None and not cascade.empty():
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.float32)

    if len(faces) == 0:
        center = (w // 2, int(h * 0.45))
        axes = (int(w * 0.32), int(h * 0.45))
    else:
        x, y, fw, fh = max(faces, key=lambda rect: rect[2] * rect[3])
        cx, cy = x + fw // 2, y + fh // 2
        axes = (int(fw * 0.75), int(fh * 0.95))
        center = (cx, cy + int(fh * 0.05))

    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, thickness=-1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    return np.clip(mask, 0.0, 1.0)


def _soften_mask(mask: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return np.clip(mask, 0.0, 1.0)
    kernel = int(max(1, round(radius)))
    kernel = kernel + 1 if kernel % 2 == 0 else kernel
    softened = cv2.GaussianBlur(mask, (kernel, kernel), radius)
    return np.clip(softened, 0.0, 1.0)


def _center_of_mask(mask: np.ndarray) -> Tuple[int, int]:
    moments = cv2.moments(mask.astype(np.float32))
    if moments["m00"] == 0:
        h, w = mask.shape
        return w // 2, h // 2
    cx = int(round(moments["m10"] / moments["m00"]))
    cy = int(round(moments["m01"] / moments["m00"]))
    return cx, cy


def _center_crop_to_aspect(image: np.ndarray, aspect: float) -> np.ndarray:
    h, w = image.shape[:2]
    current_aspect = w / h
    if abs(current_aspect - aspect) < 1e-3:
        return image.copy()

    if current_aspect > aspect:
        new_w = int(round(h * aspect))
        start = max(0, (w - new_w) // 2)
        return image[:, start:start + new_w]
    else:
        new_h = int(round(w / aspect))
        start = max(0, (h - new_h) // 2)
        return image[start:start + new_h, :]


def _align_to_mask(cat_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    cx, cy = _center_of_mask(mask)
    shift_x = cx - w / 2.0
    shift_y = cy - h / 2.0
    transformation = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    aligned = cv2.warpAffine(
        cat_image,
        transformation,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return aligned


def _match_luminance(cat_lab: np.ndarray, human_lab: np.ndarray, mask: np.ndarray) -> None:
    mask_bool = mask > 0.05
    cat_vals = cat_lab[:, :, 0][mask_bool]
    human_vals = human_lab[:, :, 0][mask_bool]
    if cat_vals.size < 32 or human_vals.size < 32:
        return
    cat_mean, cat_std = float(cat_vals.mean()), float(cat_vals.std() + 1e-6)
    human_mean, human_std = float(human_vals.mean()), float(human_vals.std() + 1e-6)
    cat_lab[:, :, 0] = (cat_lab[:, :, 0] - cat_mean) * (human_std / cat_std) + human_mean


def _blend_texture(
    human_bgr: np.ndarray,
    cat_bgr: np.ndarray,
    mask: np.ndarray,
    texture_strength: float,
    color_mix: float,
    sigma: float,
) -> np.ndarray:
    h, w = human_bgr.shape[:2]
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)

    human_lab = cv2.cvtColor(human_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Prepare and align cat texture to the facial region
    bbox = cv2.boundingRect((mask > 0.1).astype(np.uint8))
    x, y, bw, bh = bbox
    aspect = bw / bh if bh > 0 else w / h
    cat_cropped = _center_crop_to_aspect(cat_bgr, aspect)
    cat_scaled = cv2.resize(cat_cropped, (w, h), interpolation=cv2.INTER_CUBIC)
    cat_aligned = _align_to_mask(cat_scaled, mask)
    cat_lab = cv2.cvtColor(cat_aligned, cv2.COLOR_BGR2LAB).astype(np.float32)

    _match_luminance(cat_lab, human_lab, mask)

    human_low = cv2.GaussianBlur(human_lab, (0, 0), sigma)
    cat_low = cv2.GaussianBlur(cat_lab, (0, 0), sigma)
    cat_high = cat_lab - cat_low

    combined_l = human_low[:, :, 0] + texture_strength * cat_high[:, :, 0]
    combined_l = np.clip(combined_l, 0.0, 255.0)

    combined_a = (1.0 - color_mix) * human_lab[:, :, 1] + color_mix * cat_lab[:, :, 1]
    combined_b = (1.0 - color_mix) * human_lab[:, :, 2] + color_mix * cat_lab[:, :, 2]

    mask3 = mask[:, :, None]
    result_lab = human_lab.copy()
    result_lab[:, :, 0] = result_lab[:, :, 0] * (1.0 - mask) + combined_l * mask
    result_lab[:, :, 1] = result_lab[:, :, 1] * (1.0 - mask) + combined_a * mask
    result_lab[:, :, 2] = result_lab[:, :, 2] * (1.0 - mask) + combined_b * mask

    result_lab = np.clip(result_lab, 0.0, 255.0).astype(np.uint8)
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    return result_bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blend cat texture onto a human face using traditional image processing."
    )
    parser.add_argument("--human", type=Path, default=Path("4.jpg"), help="Path to human portrait image")
    parser.add_argument("--cat", type=Path, default=Path("3.jpg"), help="Path to cat texture image")
    parser.add_argument("--output", type=Path, default=Path("hybrid_result.png"), help="Output image path")
    parser.add_argument("--max-size", type=int, default=1024, help="Max longest edge for processing")
    parser.add_argument("--mask-blur", type=float, default=18.0, help="Gaussian blur radius for the face mask edges")
    parser.add_argument("--texture-strength", type=float, default=1.6, help="Strength of cat high-frequency details")
    parser.add_argument("--color-mix", type=float, default=0.65, help="Blend factor for cat colors (0=human, 1=cat)")
    parser.add_argument("--sigma", type=float, default=6.0, help="Gaussian sigma for detail separation")
    parser.add_argument(
        "--save-mask",
        type=Path,
        default=None,
        help="Optional path to save the face mask for inspection",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.human.exists() or not args.cat.exists():
        raise FileNotFoundError("Input image not found. Check --human and --cat paths.")

    human_bgr = _load_image(args.human, max_size=args.max_size)
    cat_bgr = _load_image(args.cat, max_size=args.max_size)

    mask = _detect_face_ellipse(human_bgr)
    mask = _soften_mask(mask, args.mask_blur)

    if args.save_mask is not None:
        args.save_mask.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_mask), (mask * 255).astype(np.uint8))

    result_bgr = _blend_texture(
        human_bgr,
        cat_bgr,
        mask,
        texture_strength=args.texture_strength,
        color_mix=np.clip(args.color_mix, 0.0, 1.0),
        sigma=max(0.1, args.sigma),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), result_bgr)
    print(f"Saved stylised image to {args.output}")


if __name__ == "__main__":
    main()
