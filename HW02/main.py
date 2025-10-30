import cv2
import numpy as np

from fft import fft as fft_filter
from detect import detect_bump


ROI = (531, 236, 70)


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to read image at {path}")
    return image.astype(np.float32) / 255.0


def main() -> None:
    image = load_image("source.png")
    filtered = fft_filter(image)
    position = detect_bump(filtered, ROI)
    print(f"Detected bump center: ({position[0]:.3f}, {position[1]:.3f})")


if __name__ == "__main__":
    main()
