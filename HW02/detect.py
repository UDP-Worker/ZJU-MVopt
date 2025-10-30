# detect.py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def detect_bump(img, roi, show_debug=False):
    """
    在给定圆形ROI内检测亮点（小凸起），返回其坐标（亚像素精度）。

    参数
    ----
    img : np.ndarray
        归一化灰度图像 (float32, [0,1])。
    roi : tuple
        (x_center, y_center, radius) 圆形ROI。
    show_debug : bool
        是否显示调试图像。

    返回
    ----
    (x_sub, y_sub) : tuple(float, float)
        检测到的亮点亚像素坐标。
    """
    assert img.ndim == 2, "输入必须为灰度图像"
    x_c, y_c, r = roi
    H, W = img.shape

    # Step 1. 生成圆形ROI mask
    Y, X = np.ogrid[:H, :W]
    mask = (X - x_c)**2 + (Y - y_c)**2 <= r**2
    roi_img = np.zeros_like(img)
    roi_img[mask] = img[mask]

    # Step 2. 去除大尺度背景（高斯模糊减法 = 高通滤波）
    bg = gaussian_filter(roi_img, sigma=7.0)
    hp = roi_img - bg
    hp[hp < 0] = 0.0  # 仅保留正峰值

    # Step 3. 去除暗色笔迹或阴影（可选阈值）
    thr = np.percentile(roi_img[mask], 20)  # 下20%作为阴影阈
    hp[roi_img < thr] = 0.0

    # Step 4. 在ROI内寻找最亮点
    iy, ix = np.unravel_index(np.argmax(hp), hp.shape)
    py, px = iy, ix  # 粗定位

    # Step 5. 亚像素加权质心计算
    win = 7
    wy0, wy1 = max(0, py - win//2), min(H, py + win//2 + 1)
    wx0, wx1 = max(0, px - win//2), min(W, px + win//2 + 1)
    patch = hp[wy0:wy1, wx0:wx1]
    Yg, Xg = np.mgrid[wy0:wy1, wx0:wx1]
    weights = patch - patch.min()
    weights_sum = weights.sum() + 1e-8
    y_sub = (Yg * weights).sum() / weights_sum
    x_sub = (Xg * weights).sum() / weights_sum

    # Step 6. 可视化
    if show_debug:
        fig, axes = plt.subplots(1,3, figsize=(12,4))
        axes[0].imshow(img, cmap='gray')
        circle = plt.Circle((x_c, y_c), r, color='r', fill=False, linewidth=2)
        axes[0].add_patch(circle)
        axes[0].scatter([x_sub], [y_sub], c='y', s=40)
        axes[0].set_title("Original + ROI + Detected")
        axes[0].axis('off')

        axes[1].imshow(hp, cmap='gray')
        axes[1].set_title("High-pass filtered ROI")
        axes[1].axis('off')

        axes[2].imshow(patch, cmap='gray')
        axes[2].set_title("Subpixel window")
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

        print(f"Detected bump center (subpixel): ({x_sub:.3f}, {y_sub:.3f})")

    return float(x_sub), float(y_sub)


# ---------------------- #
# 示例：直接运行测试
# ---------------------- #
if __name__ == "__main__":
    img = cv2.imread("source.png", cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0
    roi = (531, 236, 60)
    detect_bump(img, roi, show_debug=True)
