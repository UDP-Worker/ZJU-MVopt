import os
import cv2
import numpy as np

def segment_grass_sky_cloud(
        in_path="in2.png",
        out_dir="out2",
        # HSV 阈值
        grass_h_min=25, grass_h_max=95, grass_s_min=50, grass_v_min=40,
        cloud_s_max=80, cloud_v_min=120,
        grass_close_ksize=9,
        cloud_open_ksize=5
):
    os.makedirs(out_dir, exist_ok=True)

    bgr = cv2.imread(in_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"找不到图片：{in_path}（请确认与脚本同目录）")

    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    greenish = (
        (h >= grass_h_min) & (h <= grass_h_max) &
        (s >= grass_s_min) & (v >= grass_v_min)
    ).astype(np.uint8) * 255

    num, lab, stats, _ = cv2.connectedComponentsWithStats(greenish, connectivity=8)
    touch_bottom = np.zeros(num, dtype=bool)
    for lbl in range(1, num):
        y = stats[lbl, cv2.CC_STAT_TOP]
        h0 = stats[lbl, cv2.CC_STAT_HEIGHT]
        if y + h0 >= H:
            touch_bottom[lbl] = True

    grass_mask = np.isin(lab, np.where(touch_bottom)[0]).astype(np.uint8) * 255

    if grass_close_ksize and grass_close_ksize >= 3:
        k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (grass_close_ksize, grass_close_ksize)
        )
        grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_CLOSE, k_close, iterations=2)

    not_grass = cv2.bitwise_not(grass_mask)
    cloud_mask = ((s <= cloud_s_max) & (v >= cloud_v_min)).astype(np.uint8) * 255
    cloud_mask = cv2.bitwise_and(cloud_mask, not_grass)

    if cloud_open_ksize and cloud_open_ksize >= 3:
        k_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cloud_open_ksize, cloud_open_ksize)
        )
        cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_OPEN, k_open, iterations=1)

    sky_mask = cv2.bitwise_and(not_grass, cv2.bitwise_not(cloud_mask))

    grass = cv2.bitwise_and(bgr, bgr, mask=grass_mask)
    cloud = cv2.bitwise_and(bgr, bgr, mask=cloud_mask)
    sky = cv2.bitwise_and(bgr, bgr, mask=sky_mask)

    cv2.imwrite(os.path.join(out_dir, "grass.png"), grass)
    cv2.imwrite(os.path.join(out_dir, "cloud.png"), cloud)
    cv2.imwrite(os.path.join(out_dir, "sky.png"), sky)
    cv2.imwrite(os.path.join(out_dir, "grass_mask.png"), grass_mask)
    cv2.imwrite(os.path.join(out_dir, "cloud_mask.png"), cloud_mask)
    cv2.imwrite(os.path.join(out_dir, "sky_mask.png"), sky_mask)

    labels = np.zeros((H, W), dtype=np.uint8)
    labels[grass_mask == 255] = 1
    labels[sky_mask == 255] = 2
    labels[cloud_mask == 255] = 3
    cv2.imwrite(os.path.join(out_dir, "labels.png"), labels)
    cv2.imwrite(os.path.join(out_dir, "label.png"), labels * 85)

    overlay_color = np.full_like(bgr, 255)
    overlay_color[grass_mask == 255] = (0, 255, 0)
    overlay_color[sky_mask == 255] = (255, 0, 0)
    overlay_color[cloud_mask == 255] = (255, 255, 255)
    overlay = cv2.addWeighted(bgr, 0.6, overlay_color, 0.4, 0.0)
    cv2.imwrite(os.path.join(out_dir, "overlay.png"), overlay)

    print("完成。输出目录：", out_dir)
    print("像素统计：grass =", int((grass_mask == 255).sum()),
          "sky =", int((sky_mask == 255).sum()),
          "cloud =", int((cloud_mask == 255).sum()),
          "total =", H * W)

if __name__ == "__main__":
    segment_grass_sky_cloud(in_path="in2.png", out_dir="out2")
