from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.linalg import svd
from skimage.transform import rescale
from tqdm.auto import trange

from p1_utils import extract_kps_and_calc_transform, apply_transform, logger


def compute_two_boundary(H, W, a, b, c):
    x0, x1 = np.zeros_like(a), np.full_like(a, W)
    y0 = -(a * x0 + c) / b
    y1 = -(a * x1 + c) / b
    return np.stack([x0, x1]).clip(0, W - 1), np.stack([y0, y1]).clip(0, H - 1)


def display_epipolar_lines(img1_rect, img2_original, F, H1, kps, *, sample=True, k=5):
    if sample:
        idxs = np.random.choice(len(kps), size=k, replace=False)
        _, rpts = kps[idxs, 0], kps[idxs, 1]
    else:
        _, rpts = kps[:, 0], kps[:, 1]

    rpts_h = np.hstack([rpts, np.ones((len(rpts), 1))])  # shape (k, 3)

    F_rect = np.linalg.inv(H1).T @ F
    lines = (F_rect @ rpts_h.T).T  # shape (k, 3)

    x, y = compute_two_boundary(*img1_rect.shape, lines[:, 0], lines[:, 1], lines[:, 2])

    fig, axes = plt.subplots(ncols=2, figsize=(16, 5))

    cmap = plt.get_cmap("viridis")
    c = [cmap(i) for i in np.linspace(0, 1, k)]

    axes[0].imshow(img1_rect, cmap="gray")
    for i, col in enumerate(c):
        axes[0].plot(x[:, i], y[:, i], color=col)
    axes[0].axis("off")
    axes[0].set_title("Rectified Left Image with Epipolar Lines")

    axes[1].imshow(img2_original, cmap="gray")
    axes[1].scatter(rpts[:, 0], rpts[:, 1], marker="x", c=c)
    axes[1].axis("off")
    axes[1].set_title("Original Right Image with Keypoints")

    fig.suptitle("Rectified Images")
    plt.show()


def create_A(matched_points):
    ul = matched_points[:, 0, 0]
    vl = matched_points[:, 0, 1]
    ur = matched_points[:, 1, 0]
    vr = matched_points[:, 1, 1]

    ones = np.ones_like(ul)
    return np.stack([ul * ur, ul * vr, ul, vl * ur, vl * vr, vl, ur, vr, ones], axis=1)


def solve_F(A):
    # Solve Af = 0 by SVD
    U, S, Vt = svd(A)
    f = Vt[-1]
    F = f.reshape(3, 3)

    # Enforce rank-2: zero out smallest singular value
    U, S, Vt = svd(F)
    S[-1] = 0
    return U @ np.diag(S) @ Vt


def eight_point_algorithm(pts):
    pl = pts[:, 0]
    pr = pts[:, 1]

    def normalize(pts):
        mean = pts.mean(axis=0)
        shifted = pts - mean
        mean_dist = np.mean(np.sqrt((shifted**2).sum(axis=1)))
        scale = 2**0.5 / mean_dist
        T = np.array(
            [[scale, 0, -scale * mean[0]], [0, scale, -scale * mean[1]], [0, 0, 1]]
        )
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
        pts_norm_h = (T @ pts_h.T).T
        return pts_norm_h[:, :2], T

    pl_norm, T_left = normalize(pl)
    pr_norm, T_right = normalize(pr)

    matched_norm = np.stack([pl_norm, pr_norm], axis=1)
    A_norm = create_A(matched_norm)
    F_norm = solve_F(A_norm)

    # Denormalize: F = T_right^T * F_norm * T_left
    return T_right.T @ F_norm @ T_left


def rectify(image1, image2, *, logging=True, display=False):
    log = logger(logging)

    extractions = extract_kps_and_calc_transform(
        image1, image2, logging=logging, display=display
    )
    kps = extractions["keypoints"]

    log("calculate fundamental matrix F.")
    F = eight_point_algorithm(extractions["keypoints"])
    log("fundamental matrix calculated.")
    log("calculate homography matrices H1, H2.")
    _, H1, H2 = cv2.stereoRectifyUncalibrated(
        kps[:, 0],
        kps[:, 1],
        F,
        (image1.shape[1], image1.shape[0]),
    )
    log("homography matrices calculated.")
    log("apply homography matrices.")
    image1_rect = apply_transform(image1, np.linalg.inv(H1))
    image2_rect = apply_transform(image2, np.linalg.inv(H2))

    if display:
        log("display rectified images, with epipolar lines of select keypoints...")
        display_epipolar_lines(image1_rect, image2, F, H1, kps)

    return extractions | {
        "F": F,
        "H1": H1,
        "H2": H2,
        "image1_rect": image1_rect,
        "image2_rect": image2_rect,
    }


def get_cost_f(cost_f):
    def SAD(left, right):
        return np.abs(left - right).sum(axis=(2, 3))

    def SSD(left, right):
        return ((left - right) ** 2).sum(axis=(2, 3))

    def NCC(left, right):
        l = left - left.mean(axis=(2, 3), keepdims=True)
        r = right - right.mean(axis=(2, 3), keepdims=True)
        num = (l * r).sum(axis=(2, 3))
        denom = ((l * l).sum(axis=(2, 3)) * (r * r).sum(axis=(2, 3))) ** 0.5
        return -num / denom

    cost_functions = {
        "SAD": SAD,
        "SSD": SSD,
        "NCC": NCC,
    }
    cost_f = cost_functions[cost_f]
    return cost_f


def _scanline(left, right, *, cost_f="SAD", block_size=11):
    cf = get_cost_f(cost_f)

    H, W = left.shape
    bs = block_size
    dmax = W // 10

    pad = block_size // 2
    left = np.pad(left, ((pad, pad), (pad, pad)), mode="constant")
    right = np.pad(right, ((pad, pad), (pad, pad)), mode="constant")

    left_patches = sliding_window_view(left, (bs, bs))
    right_patches = sliding_window_view(right, (bs, bs))

    cost_vol = np.full((H, W, dmax), np.inf)

    for d in trange(1, dmax, desc=f"{cost_f} - {bs}"):
        rp = right_patches[:, d:]
        lp = left_patches[:, :-d or None]
        cost_vol[:, :-d or None, d] = cf(lp, rp)

    return cost_vol

def scanline(left, right, *, cost_f="SAD", block_size=11):
    return _scanline(left, right, cost_f=cost_f, block_size=block_size).argmin(axis=2)


def plot_dmaps(dmaps):
    type_count, bs_count = len(dmaps), max(len(dmaps_bs) for dmaps_bs in dmaps.values())
    fig, axes = plt.subplots(
        type_count, bs_count, figsize=(8 * bs_count, 5 * type_count)
    )
    if type_count == 1:
        axes = [axes]
    if bs_count == 1:
        axes = [[ax] for ax in axes]

    for axs, (type, dmaps_bs) in zip(axes, dmaps.items()):
        for ax, (bs, dmap) in zip(axs, dmaps_bs.items()):
            im = ax.imshow(dmap, cmap="magma")
            ax.set_title(f"{type} - {bs}")
            ax.axis("off")

            cbar = fig.colorbar(
                im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04
            )
            cbar.set_label("Disparity (px)", rotation=270, labelpad=15)
            cbar.set_ticks([dmap.min(), dmap.max()])
            cbar.set_ticklabels([dmap.min(), dmap.max()])

    plt.suptitle("Disparity Maps")
    plt.show()


def calculate_dmaps(
    image1,
    image2,
    *,
    scale=1 / 4,
    logging=True,
    display=False,
    sad=True,
    ssd=False,
    ncc=False,
    block_sizes=(5, 9, 15),
):
    log = logger(logging)

    rect_res = rectify(image1, image2, display=display)
    image1_rect, image2_rect = rect_res["image1_rect"], rect_res["image2_rect"]

    image1_rect_scal, image2_rect_scal = rescale(image1_rect, scale), rescale(
        image2_rect, scale
    )

    disp_maps = defaultdict(dict)
    if sad:
        log("calculate SAD disparity maps.")
        for bs in block_sizes:
            disp_maps["SAD"][bs] = scanline(
                image1_rect_scal, image2_rect_scal, cost_f="SAD", block_size=bs
            )
        log("SAD disparity maps calculated.")
    if ssd:
        log("calculate SSD disparity maps.")
        for bs in block_sizes:
            disp_maps["SSD"][bs] = scanline(
                image1_rect_scal, image2_rect_scal, cost_f="SSD", block_size=bs
            )
        log("SSD disparity maps calculated.")
    if ncc:
        log("calculate NCC disparity maps.")
        for bs in block_sizes:
            disp_maps["NCC"][bs] = scanline(
                image1_rect_scal, image2_rect_scal, cost_f="NCC", block_size=bs
            )
        log("NCC disparity maps calculated.")

    if display:
        log("display disparity maps...")
        plot_dmaps(disp_maps)

    return rect_res | {"disp_maps": disp_maps}
