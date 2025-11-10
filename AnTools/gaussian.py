from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np


def calculate_gaussian_radius(
    box_height: float, box_width: float, min_overlap: float = 0.7
) -> float:
    coeff_a_case1 = 1.0
    coeff_b_case1 = box_height + box_width
    coeff_c_case1 = box_width * box_height * (1 - min_overlap) / (1 + min_overlap)
    discriminant_case1 = np.sqrt(coeff_b_case1**2 - 4 * coeff_a_case1 * coeff_c_case1)
    radius_case1 = (coeff_b_case1 + discriminant_case1) / 2

    coeff_a_case2 = 4.0
    coeff_b_case2 = 2 * (box_height + box_width)
    coeff_c_case2 = (1 - min_overlap) * box_width * box_height
    discriminant_case2 = np.sqrt(coeff_b_case2**2 - 4 * coeff_a_case2 * coeff_c_case2)
    radius_case2 = (coeff_b_case2 + discriminant_case2) / 2

    coeff_a_case3 = 4 * min_overlap
    coeff_b_case3 = -2 * min_overlap * (box_height + box_width)
    coeff_c_case3 = (min_overlap - 1) * box_width * box_height
    discriminant_case3 = np.sqrt(coeff_b_case3**2 - 4 * coeff_a_case3 * coeff_c_case3)
    radius_case3 = (coeff_b_case3 + discriminant_case3) / 2
    return min(radius_case1, radius_case2, radius_case3)


def get_2d_gaussian(shape: Tuple[int, int], sigma: float = 1) -> np.ndarray:
    half_height, half_width = [(dimension - 1.0) / 2.0 for dimension in shape]
    y, x = np.ogrid[-half_height : half_height + 1, -half_width : half_width + 1]

    gaussian_kernel = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    gaussian_kernel[
        gaussian_kernel < np.finfo(gaussian_kernel.dtype).eps * gaussian_kernel.max()
    ] = 0
    return gaussian_kernel


def apply_gaussian(
    heatmap: np.ndarray,
    center: Tuple[float, float],
    radius: int,
    scale: float = 1,
) -> np.ndarray:
    diameter = 2 * radius + 1
    gaussian_kernel = get_2d_gaussian((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]
    height = int(height)
    width = int(width)

    left, right = int(min(x, radius)), int(min(width - x, radius + 1))
    top, bottom = int(min(y, radius)), int(min(height - y, radius + 1))

    radius = int(radius)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian_kernel[
        radius - top : radius + bottom, radius - left : radius + right
    ]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * scale, out=masked_heatmap)
    return heatmap


def main() -> None:
    heatmap_height = 256
    heatmap_width = 256
    center = [128, 128]
    min_overlap = 0.7
    box = [0, 0]
    scale = 1

    heatmap = np.zeros((heatmap_height, heatmap_width), dtype=np.float32)

    radius = int(np.floor(calculate_gaussian_radius(box[0], box[1], min_overlap)))
    radius = max(radius, 1)

    apply_gaussian(heatmap, (center[0], center[1]), radius, scale=scale)
    print(np.max(heatmap))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(heatmap, cmap="hot", interpolation="nearest")
    ax.scatter([center[0]], [center[1]], c="cyan", s=25, marker="x")
    ax.set_title(f"Gaussian heatmap (radius={radius})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
