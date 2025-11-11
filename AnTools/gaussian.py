from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def calculate_gaussian_radius(
    box_height: float, box_width: float, overlap: float = 0.7
) -> float:
    box_area = box_height * box_width
    gaussian_area = overlap * box_area
    radius = np.sqrt(gaussian_area / np.pi)
    return float(radius)


def get_2d_gaussian(shape: Tuple[int, int], sigma: float = 1) -> np.ndarray:
    half_height, half_width = [(dimension - 1.0) / 2.0 for dimension in shape]
    y, x = np.ogrid[-half_height : half_height + 1, -half_width : half_width + 1]

    gaussian_kernel = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    gaussian_kernel[gaussian_kernel < np.finfo(gaussian_kernel.dtype).eps * gaussian_kernel.max()] = 0
    gaussian_kernel /= gaussian_kernel.max()
    return gaussian_kernel


def apply_gaussian(
    heatmap: np.ndarray,
    center_x: float,
    center_y: float,
    radius: int,
    scale: float = 1,
) -> np.ndarray:
    diameter = 2 * radius + 1
    gaussian_kernel = get_2d_gaussian((diameter, diameter), sigma=diameter / 6)

    x, y = int(center_x), int(center_y)

    height, width = heatmap.shape[0:2]
    height, width = int(height), int(width)

    left, right = int(min(x, radius)), int(min(width - x, radius + 1))
    top, bottom = int(min(y, radius)), int(min(height - y, radius + 1))

    radius = int(radius)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian_kernel[radius - top : radius + bottom, radius - left : radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * scale, out=masked_heatmap)
    return heatmap


def main() -> None:
    heatmap_height = 144
    heatmap_width = 256
    center = [1, 87]
    overlap = 1.3
    box = [3, 15]
    scale = 1

    heatmap = np.zeros((heatmap_height, heatmap_width), dtype=np.float32)

    radius = calculate_gaussian_radius(box[0], box[1], overlap)

    apply_gaussian(heatmap, center[1], center[0], radius, scale=scale)

    _, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(heatmap, cmap="hot", interpolation="nearest")
    box_x = center[1] - box[1] / 2
    box_y = center[0] - box[0] / 2
    ax.add_patch(
        Rectangle(
            (box_x, box_y),
            box[1],
            box[0],
            linewidth=1.5,
            edgecolor="cyan",
            facecolor="none",
        )
    )
    # ax.scatter([center[1]], [center[0]], c="cyan", s=25, marker="x")
    ax.set_title(f"Gaussian heatmap (radius={radius})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
