# 高斯热图生成：目标框 → 椭圆高斯核（中心=1，边界→0）
import numpy as np


def generate_gaussian_heatmap(boxes_xyxy, out_hw, sigma_factor=0.3):
    """
    Args:
        boxes_xyxy: (K, 4) [x1,y1,x2,y2]，已缩放到 out_hw 坐标系
        out_hw:     (H, W) 输出热图尺寸
        sigma_factor: sigma = sigma_factor * (框半宽/半高)
    Returns:
        heatmap: (H, W) float32，多个框取逐点最大
    """
    H, W = out_hw
    heatmap = np.zeros((H, W), dtype=np.float32)
    if len(boxes_xyxy) == 0:
        return heatmap

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    for x1, y1, x2, y2 in boxes_xyxy:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        sx = max((x2 - x1) / 2 * sigma_factor, 1.0)
        sy = max((y2 - y1) / 2 * sigma_factor, 1.0)
        g = np.exp(-(((xx - cx) ** 2) / (2 * sx ** 2) +
                     ((yy - cy) ** 2) / (2 * sy ** 2)))
        heatmap = np.maximum(heatmap, g.astype(np.float32))
    return heatmap
