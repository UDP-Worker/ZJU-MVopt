import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk, remove_small_objects

def remove_handwriting(img, show=False):
    # 读取图像
    # img = cv2.imread("source.png", cv2.IMREAD_GRAYSCALE)
    f = img.astype(np.float32) / 255.0

    # Otsu自动阈值分割：暗区域→笔迹
    thr = threshold_otsu(f)
    ink_mask = f < thr  # True 为笔迹部分

    # 去除噪声、连通断口
    ink_mask = closing(ink_mask, disk(5))
    ink_mask = remove_small_objects(ink_mask, min_size=300)

    # 可视化笔迹掩膜
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(f, cmap='gray')
    plt.title("original image")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(ink_mask, cmap='gray')
    plt.title("auto-detected mask")
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.axis('off')

    # 掩膜覆盖后的图像（笔迹区域设为背景均值）
    masked_img = f.copy()
    masked_img[ink_mask] = np.median(f[~ink_mask])

    plt.subplot(1,3,3)
    plt.imshow(masked_img, cmap='gray')
    plt.title("image after processing")
    plt.axis('off')
    plt.axis('off')

    plt.tight_layout()
    if show:
        plt.show()

    return masked_img
if __name__ == "__main__":
    remove_handwriting(cv2.imread("source.png", cv2.IMREAD_GRAYSCALE), show=True)