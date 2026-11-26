"""Image-level operations shared by the locator pipeline and the baselines.

Coordinates are in the letterboxed (384x640) frame unless stated otherwise.
"""
import cv2
import numpy as np
import torch

IMG_SIZE = (384, 640)      # H, W — letterboxed working resolution
PATCH = 224                # candidate crop size
MIN_DIST = 32              # NMS radius (pixels) between heatmap peaks
K_PEAKS = 20               # top-K peaks kept per heatmap
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def nms_peaks(heat_np, k=K_PEAKS, min_dist=MIN_DIST):
    """heat_np (H,W) 0-1。迭代取全局最大并抑制 min_dist 半径圆盘。返回 [(x,y,val)] 降序。"""
    h = heat_np.copy()
    H, W = h.shape
    yy, xx = np.ogrid[:H, :W]
    peaks = []
    for _ in range(k):
        idx = int(np.argmax(h))
        y, x = divmod(idx, W)
        v = float(h[y, x])
        if v <= 0:
            break
        peaks.append((x, y, v))
        h[(xx - x) ** 2 + (yy - y) ** 2 < min_dist ** 2] = 0.0
    return peaks


def crop_patch(ir_raw_bgr, cx, cy):
    """从 letterboxed IR（BGR uint8）裁 224×224，越界补零。返回 RGB uint8 (224,224,3)。"""
    H, W = ir_raw_bgr.shape[:2]
    hw = PATCH // 2
    x0, y0 = int(cx) - hw, int(cy) - hw
    x1, y1 = int(cx) + hw, int(cy) + hw
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x1), min(H, y1)
    patch = np.zeros((PATCH, PATCH, 3), dtype=np.uint8)
    if x1c > x0c and y1c > y0c:
        patch[y0c - y0:y1c - y0, x0c - x0:x1c - x0] = ir_raw_bgr[y0c:y1c, x0c:x1c]
    return cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)


def norm224(img_rgb):
    """RGB uint8 224×224 → CHW ImageNet 归一化（与 _to_tensor 同口径）。"""
    t = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
    return (t - MEAN) / STD


def pck_hit(pred, gt, obj, thresh=0.5):
    """PCK@thresh·max(obj)：pred/gt 为 (x,y)，obj 为 (w,h)。"""
    return float(np.hypot(pred[0] - gt[0], pred[1] - gt[1])) <= thresh * max(obj[0], obj[1])
