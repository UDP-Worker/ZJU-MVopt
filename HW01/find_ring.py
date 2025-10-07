# find_ring_center.py
# -*- coding: utf-8 -*-
"""
从当前路径读取两幅条纹图，先做配对加性平均以抵消条纹项，随后在平均图上分割低反射圆域，
提取其边界并用代数最小二乘拟合圆，得到中心与半径；将结果叠加回两幅原图并保存。
"""
import argparse
import json
import math
import csv
from pathlib import Path
import numpy as np
import cv2


def imread_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def normalize01(im: np.ndarray) -> np.ndarray:
    im_min, im_max = float(im.min()), float(im.max())
    if im_max <= im_min:
        return np.zeros_like(im, dtype=np.float32)
    return (im - im_min) / (im_max - im_min)


def pick_center_component(mask: np.ndarray, min_area: int = 100) -> np.ndarray:
    """在二值图中挑选面积足够大且质心最接近图像中心的连通域，返回该连通域的掩膜。"""
    H, W = mask.shape
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=4
    )
    if num <= 1:
        raise RuntimeError("未检测到连通域")
    img_cy, img_cx = H / 2.0, W / 2.0

    best_id, best_d2 = None, 1e18
    for i in range(1, num):  # 0 是背景
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cy, cx = centroids[i]
        d2 = (cy - img_cy) ** 2 + (cx - img_cx) ** 2
        if d2 < best_d2:
            best_d2, best_id = d2, i

    if best_id is None:
        raise RuntimeError("没有满足面积阈值的目标连通域")
    return (labels == best_id).astype(np.uint8)


def contour_from_mask(mask: np.ndarray) -> np.ndarray:
    """从掩膜得到外边界点集（N×2，顺序无关）。"""
    # 细化为边界：形态学梯度 或 直接找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("未找到目标轮廓")
    cnt = max(contours, key=cv2.contourArea)
    pts = cnt.reshape(-1, 2).astype(np.float32)
    return pts


def fit_circle_algebraic(pts_xy: np.ndarray) -> tuple[float, float, float]:
    """
    代数最小二乘拟合：x^2 + y^2 + A x + B y + C = 0
    解 [A,B,C]^T = argmin ||M*[A,B,C]^T + b||_2
    """
    x = pts_xy[:, 0].astype(np.float64)
    y = pts_xy[:, 1].astype(np.float64)
    M = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)
    A, B, C = np.linalg.lstsq(M, b, rcond=None)[0]
    xc, yc = -A / 2.0, -B / 2.0
    r = math.sqrt(max(xc * xc + yc * yc - C, 0.0))
    return float(xc), float(yc), float(r)


def robust_refit(pts: np.ndarray, xc: float, yc: float, r: float, k: float = 2.5):
    """一次性剔除半径残差的离群点后再拟合，增强鲁棒性。"""
    rr = np.sqrt((pts[:, 0] - xc) ** 2 + (pts[:, 1] - yc) ** 2)
    resid = np.abs(rr - r)
    med = np.median(resid)
    if med <= 1e-6:
        return xc, yc, r
    keep = resid < (k * 1.4826 * med)  # MAD 规则
    xc2, yc2, r2 = fit_circle_algebraic(pts[keep])
    return xc2, yc2, r2


def overlay_and_save(img_path: Path, xc: float, yc: float, r: float, out_path: Path):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return
    # 画圆与圆心
    cv2.circle(img, (int(round(xc)), int(round(yc))), int(round(r)), (255, 255, 255), 2)
    cv2.circle(img, (int(round(xc)), int(round(yc))), 2, (255, 255, 255), 2)
    cv2.imwrite(str(out_path), img)


def main():
    ap = argparse.ArgumentParser(description="低反射圆环中心坐标估计")
    ap.add_argument("--img1", default="1.png", help="第一幅图像文件名")
    ap.add_argument("--img2", default="2.png", help="第二幅图像文件名")
    ap.add_argument("--low-quantile", type=float, default=0.20,
                    help="低反射分割的分位阈值（0~1，默认0.20表示最暗20%%）")
    ap.add_argument("--blur", type=int, default=5, help="GaussianBlur 核大小（奇数）")
    ap.add_argument("--flatfield", type=int, default=0,
                    help="是否做平坦化（大核模糊减法），0=否，>0 表示核大小")
    ap.add_argument("--min-area", type=int, default=150, help="目标连通域最小面积像素")
    args = ap.parse_args()

    p1, p2 = Path(args.img1), Path(args.img2)
    g1 = imread_gray(p1)
    g2 = imread_gray(p2)

    # 1) 配对加性平均，抵消条纹项
    avg = (g1 + g2) * 0.5

    # 可选平坦化：去除慢变化背景
    if args.flatfield and args.flatfield > 0:
        k = args.flatfield if args.flatfield % 2 == 1 else args.flatfield + 1
        bg = cv2.GaussianBlur(avg, (k, k), 0)
        avg = cv2.normalize(avg - bg, None, 0, 1, cv2.NORM_MINMAX)
        avg = (avg * 255.0).astype(np.float32)

    # 2) 轻度平滑
    k = args.blur if args.blur % 2 == 1 else args.blur + 1
    avg_blur = cv2.GaussianBlur(avg, (k, k), 0)

    # 3) 低反射区域分割（按分位数阈值）
    norm = normalize01(avg_blur)
    thresh = np.quantile(norm, args.low_quantile)
    mask = (norm < thresh).astype(np.uint8) * 255

    # 小形态学开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 4) 选择靠近图像中心的目标连通域
    target = pick_center_component(mask, min_area=args.min_area)

    # 5) 轮廓点与圆拟合（含一次鲁棒再拟合）
    pts = contour_from_mask(target)
    xc, yc, r = fit_circle_algebraic(pts)
    xc, yc, r = robust_refit(pts, xc, yc, r)

    # 6) 结果叠加并导出
    overlay_and_save(p1, xc, yc, r, Path("overlay1.png"))
    overlay_and_save(p2, xc, yc, r, Path("overlay2.png"))

    # 输出与保存
    result = {
        "center_x_cols": float(xc),
        "center_y_rows": float(yc),
        "radius_px": float(r),
        "images": [str(p1.name), str(p2.name)],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    with open("result.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_pair", "center_x(cols)", "center_y(rows)", "radius_px"])
        writer.writerow([f"{p1.name} | {p2.name}", f"{xc:.3f}", f"{yc:.3f}", f"{r:.3f}"])


if __name__ == "__main__":
    main()
