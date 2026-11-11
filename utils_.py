import json
import os
import numpy as np
from PIL import Image


def fig_rgba_u8(fig):
    # Copy the RGBA buffer to prevent later draws from mutating the snapshot.
    if not hasattr(fig.canvas, "buffer_rgba"):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        FigureCanvasAgg(fig)
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return np.ascontiguousarray(arr).copy()

def diff_to_mask(img_on_u8, img_base_u8, diff_thresh=5):
    if img_on_u8.shape != img_base_u8.shape:
        raise ValueError("Rendered images must have identical shapes.")
    if diff_thresh < 0:
        raise ValueError("diff_thresh must be nonnegative.")
    delta = np.abs(img_on_u8.astype(np.int16) - img_base_u8.astype(np.int16)).sum(axis=2)
    return delta > diff_thresh


def overlay_mask(img_rgba_u8, mask, rgb=(255, 255, 0), alpha=0.65):
    out = np.ascontiguousarray(img_rgba_u8).copy()
    c = np.array(rgb, dtype=np.float32)
    base = out[..., :3].astype(np.float32)
    base[mask] = (1 - alpha) * base[mask] + alpha * c
    out[..., :3] = np.clip(base, 0, 255).astype(np.uint8)
    return out

def mask_bbox(mask):
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def save_rgba_png(path, rgba_u8):
    arr = np.ascontiguousarray(rgba_u8).copy()
    Image.fromarray(arr, mode="RGBA").save(path)


def save_gray_png(path, gray_u8):
    arr = np.ascontiguousarray(gray_u8).copy()
    Image.fromarray(arr, mode="L").save(path)
