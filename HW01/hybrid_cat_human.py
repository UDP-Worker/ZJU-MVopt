import argparse
import math
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple
from urllib.error import URLError

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import models, transforms


_DEFAULT_TORCH_HOME = Path(__file__).resolve().parent / ".torch_cache"
os.environ.setdefault("TORCH_HOME", str(_DEFAULT_TORCH_HOME))
_DEFAULT_TORCH_HOME.mkdir(parents=True, exist_ok=True)


def _load_image(path: Path, device: torch.device, max_size: int = 512) -> Tuple[torch.Tensor, Image.Image]:
    """Load an image as a 0-1 float tensor and resize to respect max dimension."""
    image = Image.open(path).convert("RGB")
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / float(max(w, h))
        new_size = (int(round(w * scale)), int(round(h * scale)))
        image = image.resize(new_size, Image.LANCZOS)
    tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    return tensor, image


def _detect_face_ellipse(image: Image.Image) -> np.ndarray:
    """Detect the main face and return a soft elliptical mask (numpy 2D array)."""
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    data_root = getattr(getattr(cv2, "data", object()), "haarcascades", None)
    if data_root is None:
        cv2_base = Path(cv2.__file__).resolve().parent
        data_root = cv2_base / "data"
    cascade_path = str(Path(data_root) / "haarcascade_frontalface_default.xml")
    faces = ()
    if Path(cascade_path).exists():
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if not face_cascade.empty():
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.float32)

    if len(faces) == 0:
        # Fallback ellipse roughly covering facial area
        center = (w // 2, int(h * 0.45))
        axes = (int(w * 0.32), int(h * 0.45))
    else:
        # Take the largest detected face
        x, y, fw, fh = max(faces, key=lambda rect: rect[2] * rect[3])
        cx, cy = x + fw // 2, y + fh // 2
        # Expand ellipse slightly to include hairline/cheeks
        axes = (int(fw * 0.75), int(fh * 0.95))
        center = (cx, cy + int(fh * 0.05))

    cv2.ellipse(mask, center, axes, angle=0, startAngle=0, endAngle=360, color=1.0, thickness=-1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    mask = np.clip(mask, 0.0, 1.0)
    if mask.max() > 0:
        mask /= mask.max()
    return mask


def _prepare_mask(image_tensor: torch.Tensor) -> torch.Tensor:
    """Create a face mask aligned with the tensor's spatial size."""
    pil_image = transforms.ToPILImage()(image_tensor.squeeze(0).cpu())
    mask_np = _detect_face_ellipse(pil_image)
    mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
    return mask.to(image_tensor.device, image_tensor.dtype)


def _gram_matrix(features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    b, c, h, w = features.size()
    feat = features.view(b, c, h * w)

    if mask is None:
        gram = torch.bmm(feat, feat.transpose(1, 2))
        return gram / (c * h * w)

    mask_resized = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
    mask_flat = mask_resized.view(b, 1, h * w)
    weighted = feat * mask_flat
    mask_sum = mask_flat.sum(dim=2, keepdim=True).clamp_min(1e-8)
    scale = (h * w) / mask_sum
    gram = torch.bmm(weighted, weighted.transpose(1, 2))
    gram = gram * scale / (c * h * w)
    return gram


def _get_features(
    model: nn.Sequential,
    x: torch.Tensor,
    layers: Iterable[str],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Run VGG features and gather intermediate outputs."""
    normalized = (x - mean) / std
    target_layers = tuple(layers)
    target_set = set(target_layers)
    target_count = len(target_layers)
    feats: Dict[str, torch.Tensor] = {}
    current = normalized
    for name, layer in model._modules.items():
        current = layer(current)
        if name in target_set:
            feats[name] = current
            if len(feats) == target_count:
                break
    return feats


def _run_style_transfer(
    content: torch.Tensor,
    style: torch.Tensor,
    content_mask: torch.Tensor,
    style_mask: torch.Tensor,
    device: torch.device,
    num_steps: int = 400,
    style_weight: float = 1e4,
    content_weight: float = 1.0,
    pixel_preserve_weight: float = 30.0,
    weights_path: Path | None = None,
) -> torch.Tensor:
    weights = models.VGG19_Weights.IMAGENET1K_V1
    if weights_path is not None:
        if not weights_path.exists():
            raise FileNotFoundError(f"Provided VGG weights not found: {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu")
        backbone = models.vgg19(weights=None)
        missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Failed to load VGG19 weights from provided file. "
                f"Missing keys: {missing}, unexpected keys: {unexpected}"
            )
    else:
        try:
            backbone = models.vgg19(weights=weights)
        except URLError as err:
            raise RuntimeError(
                "Unable to download VGG19 weights automatically. "
                "Please download the file manually and pass --vgg-weights /path/to/vgg19-dcbb9e9d.pth"
            ) from err

    vgg = backbone.features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False

    transform_cfg = weights.transforms()
    mean = torch.tensor(transform_cfg.mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(transform_cfg.std, device=device).view(1, 3, 1, 1)

    content_layers = ["21"]  # relu4_2
    style_layers = ["0", "5", "10", "19", "28"]  # relu1_1, relu2_1, ..., relu5_1
    style_layer_weights = {
        "0": 0.2,
        "5": 0.2,
        "10": 0.2,
        "19": 0.2,
        "28": 0.2,
    }

    content_targets = _get_features(vgg, content, content_layers, mean, std)
    style_targets = _get_features(vgg, style, style_layers, mean, std)
    style_grams = {
        layer: _gram_matrix(feat, mask=style_mask).detach() for layer, feat in style_targets.items()
    }

    input_img = content.clone().requires_grad_(True)
    optimizer = optim.Adam([input_img], lr=0.03)

    background_mask = 1.0 - content_mask

    combined_layers = tuple(dict.fromkeys(content_layers + style_layers))

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()

        input_features = _get_features(vgg, input_img, combined_layers, mean, std)

        content_loss = 0.0
        for layer in content_layers:
            content_loss = content_loss + F.mse_loss(input_features[layer], content_targets[layer])

        style_loss = 0.0
        for layer in style_layers:
            input_gram = _gram_matrix(input_features[layer], mask=content_mask)
            target_gram = style_grams[layer]
            style_loss = style_loss + style_layer_weights[layer] * F.mse_loss(input_gram, target_gram)

        pixel_loss = F.mse_loss(input_img * background_mask, content * background_mask)

        total_loss = content_weight * content_loss + style_weight * style_loss + pixel_preserve_weight * pixel_loss
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            input_img.clamp_(0.0, 1.0)

        if step % 50 == 0 or step == 1:
            print(
                f"Step {step}/{num_steps}: total={total_loss.item():.2f}, "
                f"content={content_loss.item():.2f}, style={style_loss.item():.2f}, pixel={pixel_loss.item():.2f}"
            )

    return input_img.detach()


def _save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    image = transforms.ToPILImage()(tensor.squeeze(0).cpu())
    image.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neural style transfer with elliptical face constraint")
    parser.add_argument("--content", type=Path, default=Path("4.jpg"), help="Path to human portrait image")
    parser.add_argument("--style", type=Path, default=Path("3.jpg"), help="Path to cat texture image")
    parser.add_argument("--output", type=Path, default=Path("hybrid_result.png"), help="Where to save the output image")
    parser.add_argument("--max-size", type=int, default=512, help="Max size (longest edge) for processing")
    parser.add_argument("--steps", type=int, default=400, help="Number of optimization steps")
    parser.add_argument("--style-weight", type=float, default=1e3, help="Weight for style loss")
    parser.add_argument("--content-weight", type=float, default=1.0, help="Weight for content loss")
    parser.add_argument(
        "--pixel-preserve-weight",
        type=float,
        default=50.0,
        help="Weight to keep background close to original content",
    )
    parser.add_argument(
        "--vgg-weights",
        type=Path,
        default=None,
        help="Optional local path to vgg19-dcbb9e9d.pth to avoid downloads",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.content.exists() or not args.style.exists():
        raise FileNotFoundError("Content or style image not found. Please check the provided paths.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    content_tensor, _ = _load_image(args.content, device=device, max_size=args.max_size)
    style_tensor, _ = _load_image(args.style, device=device, max_size=args.max_size)

    content_mask = _prepare_mask(content_tensor)
    style_mask = _prepare_mask(style_tensor)
    if torch.cuda.is_available():
        content_mask = content_mask.to(device)
        style_mask = style_mask.to(device)
    content_mask = content_mask.clamp(0.0, 1.0)
    style_mask = style_mask.clamp(0.0, 1.0)

    print("Face mask coverage: {:.2f}%".format(content_mask.mean().item() * 100))

    output = _run_style_transfer(
        content_tensor,
        style_tensor,
        content_mask,
        style_mask,
        device=device,
        num_steps=args.steps,
        style_weight=args.style_weight,
        content_weight=args.content_weight,
        pixel_preserve_weight=args.pixel_preserve_weight,
        weights_path=args.vgg_weights,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _save_tensor_image(output, args.output)
    print(f"Saved stylized image to {args.output}")


if __name__ == "__main__":
    main()
