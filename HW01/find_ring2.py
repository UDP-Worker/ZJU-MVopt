# pipeline_debug.py
import json, csv
from pathlib import Path
import numpy as np
import cv2

def imread_gray(p: str) -> np.ndarray:
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(p)
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im.astype(np.float32)

def save_gray01(name: str, im01: np.ndarray):
    im8 = (np.clip(im01, 0.0, 1.0) * 255).astype(np.uint8)
    cv2.imwrite(name, im8)

def normalize01(im: np.ndarray) -> np.ndarray:
    mn, mx = float(im.min()), float(im.max())
    if mx <= mn:
        return np.zeros_like(im, np.float32)
    return (im - mn) / (mx - mn)

def average_pair(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    return (im1 + im2) * 0.5

def box_blur_integral(im: np.ndarray, k: int) -> np.ndarray:
    assert im.ndim == 2 and k % 2 == 1
    p = k // 2
    im_pad = np.pad(im, ((p, p), (p, p)), mode='reflect')
    ii = im_pad.astype(np.float64).cumsum(0).cumsum(1)
    ii = np.pad(ii, ((1, 0), (1, 0)), mode='constant', constant_values=0)
    K = k
    sums = ii[K:, K:] - ii[:-K, K:] - ii[K:, :-K] + ii[:-K, :-K]
    return (sums / (K * K)).astype(np.float32)

def conv_sum2d01(mask01: np.ndarray, k: int = 3) -> np.ndarray:
    p = k // 2
    pad = np.pad(mask01, ((p, p), (p, p)), mode='reflect')
    ii = pad.cumsum(0).cumsum(1)
    ii = np.pad(ii, ((1, 0), (1, 0)), mode='constant', constant_values=0)
    K = k
    return ii[K:, K:] - ii[:-K, K:] - ii[K:, :-K] + ii[:-K, :-K]

def morph_open3(mask01: np.ndarray) -> np.ndarray:
    cnt = conv_sum2d01(mask01, 3)
    eroded = (cnt == 9).astype(np.uint8)
    cnt2 = conv_sum2d01(eroded, 3)
    dil = (cnt2 > 0).astype(np.uint8)
    return dil

def boundary_from_mask(mask01: np.ndarray) -> np.ndarray:
    cnt = conv_sum2d01(mask01, 3)
    boundary = (mask01 == 1) & (cnt < 9)
    ys, xs = np.nonzero(boundary)
    return np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)

def fit_circle_algebraic(pts_xy: np.ndarray):
    x, y = pts_xy[:, 0], pts_xy[:, 1]
    M = np.column_stack([x, y, np.ones_like(x)])
    b = -(x*x + y*y)
    A, B, C = np.linalg.lstsq(M, b, rcond=None)[0]
    xc, yc = -A / 2.0, -B / 2.0
    r = float(np.sqrt(max(xc*xc + yc*yc - C, 0.0)))
    return float(xc), float(yc), r

def pick_center_component(mask01: np.ndarray, min_area: int = 150) -> np.ndarray:
    H, W = mask01.shape
    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask01.astype(np.uint8), connectivity=4)
    if num <= 1:
        return mask01 * 0
    img_cy, img_cx = H/2.0, W/2.0

    best, bestd = None, 1e18
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cy, cx = cents[i]
        d2 = (cy - img_cy)**2 + (cx - img_cx)**2
        if d2 < bestd:
            best, bestd = i, d2
    if best is None:
        for i in range(1, num):
            cy, cx = cents[i]
            d2 = (cy - img_cy)**2 + (cx - img_cx)**2
            if d2 < bestd:
                best, bestd = i, d2
    return (labels == best).astype(np.uint8), stats, cents

def run_pipeline(img1_path="1.png", img2_path="2.png",
                 k_bg=51,         # 大核背景（平坦化）
                 k_smooth=5,      # 小核去噪
                 dark_offset=0.05,# 相对背景的暗度阈值（越大越严格）
                 min_area=2000    # 目标最小面积（按你的图像分辨率可调）
                 ):
    out_dir = Path(".")
    # Step 0: 读入与归一化
    im1 = normalize01(imread_gray(img1_path))
    im2 = normalize01(imread_gray(img2_path))
    save_gray01(str(out_dir/"step0_img1_norm.png"), im1)
    save_gray01(str(out_dir/"step0_img2_norm.png"), im2)

    # Step 1: 配对平均
    avg = average_pair(im1, im2)
    save_gray01(str(out_dir/"step1_avg.png"), avg)

    # Step 2: 平坦化
    bg = box_blur_integral(avg, k=k_bg)
    save_gray01(str(out_dir/"step2_bg_box{:d}.png".format(k_bg)), normalize01(bg))
    rel = normalize01(bg - avg)
    save_gray01(str(out_dir/"step2_rel_bg_minus_avg.png"), rel)

    # Step 3: 小核去噪
    blur = box_blur_integral(rel, k=k_smooth)
    save_gray01(str(out_dir/"step3_rel_box{:d}.png".format(k_smooth)), blur)

    # Step 4: 阈值分割（相对背景更“暗”的区域）
    med = float(np.median(blur))
    mad = float(np.median(np.abs(blur - med))) + 1e-6
    T = max(med + 2.5*mad, med + dark_offset)  # 你可以把 2.5 改小/改大
    mask = (blur > T).astype(np.uint8)  # rel=bg-avg 越大越暗
    save_gray01(str(out_dir/"step4_mask_raw.png"), mask)

    # Step 5: 3x3 开运算，去掉细小条纹与噪点
    mask_open = morph_open3(mask)
    save_gray01(str(out_dir/"step5_mask_open.png"), mask_open)

    # Step 6: 连通域标注与可视化
    comp, stats, cents = pick_center_component(mask_open, min_area=min_area)
    num, labels = cv2.connectedComponents(mask_open.astype(np.uint8))
    vis_lab = (255.0 * labels / max(1, num-1)).astype(np.uint8)
    cv2.imwrite(str(out_dir/"step6_labels.png"), vis_lab)
    save_gray01(str(out_dir/"step6_target_mask.png"), comp)

    # Step 7: 取边界并拟合圆
    pts = boundary_from_mask(comp)
    if pts.shape[0] < 10:
        raise RuntimeError("边界点过少，阈值/面积阈值可能过严。请调小 dark_offset 或 min_area。")
    xc, yc, r = fit_circle_algebraic(pts)

    # Step 8: 可视化叠加
    base8 = (avg * 255).clip(0, 255).astype(np.uint8)
    base3 = cv2.cvtColor(base8, cv2.COLOR_GRAY2BGR)
    for x, y in pts[:: max(1, pts.shape[0]//500) ]:
        cv2.circle(base3, (int(round(x)), int(round(y))), 0, (255, 255, 255), 1)
    cv2.circle(base3, (int(round(xc)), int(round(yc))), int(round(r)), (255, 255, 255), 2)
    cv2.circle(base3, (int(round(xc)), int(round(yc))), 2, (255, 255, 255), 2)
    cv2.imwrite(str(out_dir/"step8_boundary_fit_overlay.png"), base3)

    # 结果输出
    result = {
        "center_x(cols)": round(float(xc), 3),
        "center_y(rows)": round(float(yc), 3),
        "radius_px": round(float(r), 3),
        "threshold_T": round(T, 6),
        "median": round(med, 6),
        "mad": round(mad, 6),
        "params": {"k_bg": k_bg, "k_smooth": k_smooth,
                   "dark_offset": dark_offset, "min_area": min_area}
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    Path("result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with open("result.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f);
        w.writerow(["center_x(cols)", "center_y(rows)", "radius_px", "T", "median", "mad", "k_bg", "k_smooth", "dark_offset", "min_area"])
        w.writerow([result["center_x(cols)"], result["center_y(rows)"], result["radius_px"],
                    result["threshold_T"], result["median"], result["mad"],
                    k_bg, k_smooth, dark_offset, min_area])

if __name__ == "__main__":
    run_pipeline(
        img1_path="1.png",
        img2_path="2.png",
        k_bg=51,          # 背景核,去除条纹残留
        k_smooth=5,       # 细节降噪
        dark_offset=0.05, # 最小暗度偏移阈值（0.03~0.10 可调）
        min_area=2000     # 目标连通域的最小面积（分辨率不同需调整）
    )
