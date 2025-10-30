import cv2
import numpy as np
from skimage.feature import peak_local_max

def fft(
        img: np.ndarray,
        threshold_rel: float = 0.15,   # 峰检测阈值（越小→去网格更强）
        min_distance: int = 12,        # 峰间最小距离
        dc_margin: int = 25,           # 屏蔽中心直流区的半径
        band: int = 30,                # 限制峰靠近水平/垂直方向
        sigma: float = 15.0,            # 高斯陷波半径（越大→去网格更彻底，但可能模糊）
        num_peaks: int = 24,           # 最大峰数量
        restrict_axes: bool = False,    # 是否仅保留水平/垂直方向峰
        normalize_output: bool = True, # 是否输出归一化到[0,1]的结果
        show_debug: bool = False       # 是否显示调试图
) -> np.ndarray:
    """
    基于FFT的网格去除滤波器。
    输入与输出均为float32, [0,1]的灰度图。
    """
    assert img.ndim == 2, "img必须是灰度图像 (H, W)。"
    img = img.astype(np.float32)
    H, W = img.shape

    # Step 1. FFT + 移频
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)

    # Step 2. 计算频谱对数幅度
    mag = np.log1p(np.abs(Fshift))
    mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-12)

    # Step 3. 屏蔽直流中心区域
    allow = np.ones_like(mag_norm, dtype=bool)
    r0, r1 = H//2 - dc_margin, H//2 + dc_margin + 1
    c0, c1 = W//2 - dc_margin, W//2 + dc_margin + 1
    allow[r0:r1, c0:c1] = False
    mag_for_peaks = mag_norm.copy()
    mag_for_peaks[~allow] = 0.0

    # Step 4. 自动寻找峰
    peaks = peak_local_max(
        mag_for_peaks,
        min_distance=int(min_distance),
        threshold_rel=float(threshold_rel),
        num_peaks=int(num_peaks)
    )
    sel = []
    if restrict_axes:
        for r, c in peaks:
            if abs(r - H//2) < band or abs(c - W//2) < band:
                sel.append((int(r), int(c)))
    else:
        sel = [(int(r), int(c)) for r, c in peaks]

    # Step 5. 对称点补全
    sel_sym = set()
    for r, c in sel:
        sel_sym.add((r, c))
        sel_sym.add((H - 1 - r, W - 1 - c))
    sel = sorted(list(sel_sym))

    # Step 6. 生成高斯陷波掩膜
    rr, cc = np.ogrid[:H, :W]
    notch_mask = np.ones((H, W), dtype=np.float32)
    s2 = sigma**2 + 1e-12
    for (r, c) in sel:
        d2 = (rr - r)**2 + (cc - c)**2
        notch_mask *= 1.0 - np.exp(-0.5 * d2 / s2)

    # Step 7. 频域滤波 + 逆变换
    Fshift_filt = Fshift * notch_mask
    img_filt = np.fft.ifft2(np.fft.ifftshift(Fshift_filt))
    img_filt = np.real(img_filt).astype(np.float32)

    # Step 8. 归一化输出
    if normalize_output:
        mn, mx = img_filt.min(), img_filt.max()
        img_out = (img_filt - mn) / (mx - mn + 1e-12)
    else:
        img_out = img_filt

    # img_out = 1 - img_out

    # Step 9. 可选可视化
    if show_debug:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.imshow(img, cmap='gray')
        plt.title("Original")
        plt.axis('off')

        plt.subplot(1,3,2)
        plt.imshow(np.log1p(np.abs(Fshift)), cmap='gray')
        plt.title("Log Spectrum")
        plt.axis('off')

        plt.subplot(1,3,3)
        plt.imshow(img_out, cmap='gray')
        plt.title("Filtered")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        print("Detected notch centers:", sel)

    return img_out

if __name__ == "__main__":
    fft(cv2.imread("source.png", cv2.IMREAD_GRAYSCALE), show_debug=True)