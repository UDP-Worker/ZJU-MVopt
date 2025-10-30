import json

import cv2
import numpy as np
import matplotlib.pyplot as plt

from fft import fft as fft_filter
from detect import detect_bump


ROI = (540, 220, 80)
OUTPUT_IMAGE_ORIGINAL = "detection_result.png"
OUTPUT_IMAGE_FILTERED = "detection_result_filtered.png"
DETECTION_COORDS_PATH = "detection_coordinates.json"


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to read image at {path}")
    return image.astype(np.float32) / 255.0


def visualize_detection(
        image: np.ndarray,
        position: tuple[float, float],
        output_path: str,
        title: str
) -> None:
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    ax.scatter([position[0]], [position[1]], s=60, c="yellow", edgecolors="black", linewidths=1.0)
    circle = plt.Circle((ROI[0], ROI[1]), ROI[2], color="red", fill=False, linewidth=1.5)
    ax.add_patch(circle)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()
    plt.close(fig)


def save_coordinates(path: str, position: tuple[float, float]) -> None:
    data = {"x": float(position[0]), "y": float(position[1])}
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=True, indent=2)


def main() -> None:
    image = load_image("source.png")
    filtered = fft_filter(image)
    position = detect_bump(filtered, ROI)
    print(f"Detected bump center: ({position[0]:.3f}, {position[1]:.3f})")
    save_coordinates(DETECTION_COORDS_PATH, position)
    visualize_detection(image, position, OUTPUT_IMAGE_ORIGINAL, "Detected bump on original image")
    visualize_detection(filtered, position, OUTPUT_IMAGE_FILTERED, "Detected bump on FFT-filtered image")


if __name__ == "__main__":
    main()
