"""Optional image-difference visualization using NumPy and Pillow."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def draw_grid(image, grid_size=28, color=(0, 255, 0), thickness=1):
    """Draw an RGB grid in place and return the input array."""
    if grid_size <= 0 or thickness <= 0:
        raise ValueError("Grid size and thickness must be positive.")
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    height, width = image.shape[:2]
    for x in range(0, width, grid_size):
        draw.line((x, 0, x, height), fill=color, width=thickness)
    for y in range(0, height, grid_size):
        draw.line((0, y, width, y), fill=color, width=thickness)
    image[...] = np.asarray(canvas)
    return image


def highlight_grids(original_img, diff_img, grid_size=28,
                    highlight_color=(255, 255, 0), fade_alpha=0.3):
    """Highlight changed grid cells, fading the remaining cells."""
    if grid_size <= 0 or not 0 <= fade_alpha <= 1:
        raise ValueError("Grid size must be positive and fade_alpha must be in [0, 1].")
    if original_img.shape[:2] != diff_img.shape[:2]:
        raise ValueError("Image dimensions must match.")
    result = (original_img.astype(float) * fade_alpha).astype(np.uint8)
    height, width = original_img.shape[:2]
    color = np.asarray(highlight_color, dtype=float)
    for y in range(0, height, grid_size):
        for x in range(0, width, grid_size):
            cell = np.s_[y:y + grid_size, x:x + grid_size]
            if np.any(diff_img[cell]):
                result[cell] = np.clip(0.7 * original_img[cell] + 0.3 * color, 0, 255).astype(np.uint8)
    return result


def image_difference(image1_path, image2_path, output_path="difference.png"):
    """Save differences without resizing inputs; return difference and two RGB grid views."""
    with Image.open(image1_path) as image:
        first = np.asarray(image.convert("RGB"))
    with Image.open(image2_path) as image:
        second = np.asarray(image.convert("RGB"))
    if first.shape != second.shape:
        raise ValueError("Input images must have identical dimensions.")
    difference = np.abs(first.astype(np.int16) - second.astype(np.int16)).astype(np.uint8)
    original_grid = draw_grid(first.copy())
    diff_grid = draw_grid(difference.copy(), color=(255, 0, 0))
    highlighted = draw_grid(highlight_grids(first, difference))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    for path, data in [(output, difference), (output.with_name("image1_with_grid.png"), original_grid),
                       (output.with_name("diff_with_grid.png"), diff_grid),
                       (output.with_name("highlighted_image1.png"), highlighted)]:
        Image.fromarray(data).save(path)
    return difference, original_grid, highlighted


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--output", default="difference.png")
    args = parser.parse_args()
    image_difference(args.first, args.second, args.output)
