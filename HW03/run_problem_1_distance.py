import os
import math
import cv2
import numpy as np

def _features_from_component(comp_u8: np.ndarray):
    cnts, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0, 0.0, 0.0
    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    peri = float(cv2.arcLength(cnt, True))
    circularity = 0.0 if peri <= 0 else 4.0 * math.pi * area / (peri * peri)

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = 0.0 if hull_area <= 0 else area / hull_area
    return area, circularity, solidity

def _count_distance_peaks(comp_u8: np.ndarray, win: int, min_val: float):
    dist = cv2.distanceTransform(comp_u8, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return 0, dist, np.zeros_like(comp_u8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (win, win))
    dil = cv2.dilate(dist, k)
    maxima = ((dist == dil) & (dist >= min_val)).astype(np.uint8) * 255

    core = cv2.erode(comp_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    maxima = cv2.bitwise_and(maxima, maxima, mask=core)

    n, _ = cv2.connectedComponents(maxima)
    return n - 1, dist, maxima

def classify_particles_v2(
        in_path="in.png",
        out_dir="out",
        opening_ksize=3,
        # 面积判据
        overlap_area_factor=1.3,
        # 距离峰值阈值系数
        peak_min_rel=0.8,
        # 峰值检测窗口系数，约等于颗粒直径量级即可
        peak_win_rel=0.9,
        # 形状辅助判据
        circularity_thr=0.80,
        solidity_thr=0.97,
        debug_save_peaks=True
):
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"找不到图片：{in_path}")

    H, W = img.shape

    # 二值化，黑颗粒作为前景
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 去噪
    if opening_ksize and opening_ksize >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_ksize, opening_ksize))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)

    areas = stats[1:, cv2.CC_STAT_AREA]
    xs = stats[1:, cv2.CC_STAT_LEFT]
    ys = stats[1:, cv2.CC_STAT_TOP]
    ws = stats[1:, cv2.CC_STAT_WIDTH]
    hs = stats[1:, cv2.CC_STAT_HEIGHT]

    touch_border = (xs == 0) | (ys == 0) | (xs + ws >= W) | (ys + hs >= H)

    # 估计单颗粒面积 A0，从而得到半径的估计值
    areas_nb = areas[~touch_border]
    if len(areas_nb) == 0:
        raise RuntimeError("所有联通分量都触边，无法估计单颗粒尺寸。")

    med = float(np.median(areas_nb))
    single_candidates = areas_nb[areas_nb <= med]
    A0 = float(np.median(single_candidates)) if len(single_candidates) > 0 else med
    r0 = math.sqrt(A0 / math.pi)

    # 峰值检测参数
    win = int(round(r0 * peak_win_rel))
    if win % 2 == 0:
        win += 1
    win = max(5, win)
    min_val = peak_min_rel * r0

    # 输出掩膜
    mask_incomplete = np.zeros((H, W), dtype=np.uint8)
    mask_overlap = np.zeros((H, W), dtype=np.uint8)
    mask_complete = np.zeros((H, W), dtype=np.uint8)

    rows = []

    for lbl in range(1, num_labels):
        x, y, w, h, area_px = stats[lbl]
        cx, cy = centroids[lbl]
        comp = (labels == lbl).astype(np.uint8) * 255

        # 类别：1 不完整，2 重叠，3 完整独立
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            cls = 1
            mask_incomplete[labels == lbl] = 255
            peaks = 0
            circularity = 0.0
            solidity = 0.0
        else:
            peaks, dist, peaks_mask = _count_distance_peaks(comp, win=win, min_val=min_val)
            _, circularity, solidity = _features_from_component(comp)

            # 核心判据：峰值数
            is_overlap = (peaks >= 2)

            # 辅助判据：面积明显偏大或者似圆度偏低
            if (area_px >= overlap_area_factor * A0) or (solidity > 0 and solidity < solidity_thr) or (circularity > 0 and circularity < circularity_thr):
                is_overlap = True

            if is_overlap:
                cls = 2
                mask_overlap[labels == lbl] = 255
            else:
                cls = 3
                mask_complete[labels == lbl] = 255

            if debug_save_peaks and peaks_mask is not None and peaks > 0:
                cv2.imwrite(os.path.join(out_dir, f"peaks_{lbl:03d}.png"), peaks_mask)

        rows.append([lbl, cls, int(area_px), float(cx), float(cy), int(x), int(y), int(w), int(h), int(peaks), float(circularity), float(solidity)])

    # 叠加可视化
    color = np.full((H, W, 3), 255, dtype=np.uint8)
    color[mask_incomplete == 255] = (255, 0, 0)   # 蓝：不完整
    color[mask_overlap == 255] = (0, 0, 255)      # 红：重叠
    color[mask_complete == 255] = (0, 255, 0)     # 绿：完整

    orig_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(orig_bgr, 0.6, color, 0.4, 0.0)

    cv2.imwrite(os.path.join(out_dir, "mask_incomplete.png"), mask_incomplete)
    cv2.imwrite(os.path.join(out_dir, "mask_overlap.png"), mask_overlap)
    cv2.imwrite(os.path.join(out_dir, "mask_complete.png"), mask_complete)
    cv2.imwrite(os.path.join(out_dir, "overlay.png"), overlay)

    csv_path = os.path.join(out_dir, "particles.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("label,class,area,cx,cy,x,y,w,h,peaks,circularity,solidity\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

    n1 = sum(1 for r in rows if r[1] == 1)
    n2 = sum(1 for r in rows if r[1] == 2)
    n3 = sum(1 for r in rows if r[1] == 3)

    print(f"完成。估计单颗粒面积 A0≈{A0:.1f} 像素，等效半径 r0≈{r0:.2f} 像素")
    print(f"峰值窗 win={win}，峰值阈值 min_val≈{min_val:.2f}")
    print(f"不完整={n1} 重叠={n2} 完整独立={n3}")
    print(f"输出目录：{out_dir}")

if __name__ == "__main__":
    classify_particles_v2(in_path="in1.png", out_dir="out1")
