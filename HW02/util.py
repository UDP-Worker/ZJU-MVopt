# util.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

def select_circle_roi_interactive(img):
    """
    交互式选择圆形 ROI，返回 (x_center, y_center, radius)。

    参数
    ----
    img : np.ndarray
        输入灰度或彩色图像，类型 float32 或 uint8。

    返回
    ----
    roi_circle : tuple
        (x_center, y_center, radius)，均为整数像素坐标。
    """
    roi = {}
    clicks = []

    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
    ax.set_title("请依次点击：圆心 → 圆周上一点")
    plt.axis("off")

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        clicks.append((int(event.xdata), int(event.ydata)))
        if len(clicks) == 1:
            ax.plot(event.xdata, event.ydata, 'ro')
            fig.canvas.draw()
        elif len(clicks) == 2:
            # 计算圆心与半径
            (x0, y0), (x1, y1) = clicks
            r = int(np.hypot(x1 - x0, y1 - y0))
            roi["circle"] = (x0, y0, r)
            circle = plt.Circle((x0, y0), r, color='r', fill=False, linewidth=2)
            ax.add_patch(circle)
            plt.title(f"ROI确定: 圆心=({x0},{y0}), 半径={r}")
            fig.canvas.draw()
            print(f"已选择圆形ROI: center=({x0}, {y0}), radius={r}")
            plt.close(fig)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    if "circle" not in roi:
        raise RuntimeError("未选择ROI或窗口被关闭。")
    return roi["circle"]

def draw_circle_roi(img, roi, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制圆形 ROI。

    参数
    ----
    img : np.ndarray
        输入图像。
    roi : tuple
        (x_center, y_center, radius)
    color : tuple
        边框颜色 (B, G, R)
    thickness : int
        线宽
    """
    x, y, r = roi
    img_out = img.copy()
    cv2.circle(img_out, (int(x), int(y)), int(r), color, thickness)
    return img_out

def extract_circle_roi(img, roi):
    """
    根据圆形ROI提取图像中的区域，外部像素置零。

    参数
    ----
    img : np.ndarray
        输入图像。
    roi : tuple
        (x_center, y_center, radius)
    """
    x, y, r = roi
    Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
    mask = (X - x)**2 + (Y - y)**2 <= r**2
    roi_img = np.zeros_like(img)
    roi_img[mask] = img[mask]
    return roi_img

# --------------------- #
# 示例：直接运行脚本测试
# --------------------- #
if __name__ == "__main__":
    img = cv2.imread("source.png", cv2.IMREAD_GRAYSCALE)
    img = img.astype("float32") / 255.0
    roi = select_circle_roi_interactive(img)
    img_marked = draw_circle_roi((img*255).astype("uint8"), roi)

    plt.imshow(img_marked, cmap='gray')
    plt.title("带圆形ROI标记的图像")
    plt.axis('off')
    plt.show()
    print(roi)
