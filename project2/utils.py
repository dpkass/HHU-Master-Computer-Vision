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
        mean_dist = np.mean(np.sqrt((shifted ** 2).sum(axis=1)))
        scale = 2 ** 0.5 / mean_dist
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


def rectify(left, right, *, logging=True, display=False):
    log = logger(logging)

    extractions = extract_kps_and_calc_transform(
        left, right, logging=logging, display=display
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
        (left.shape[1], left.shape[0]),
    )
    log("homography matrices calculated.")
    log("apply homography matrices.")
    ones = np.ones_like(left, dtype=np.bool)
    left_rect = apply_transform(left, np.linalg.inv(H1))
    left_mask = apply_transform(ones, np.linalg.inv(H1))
    right_rect = apply_transform(right, np.linalg.inv(H2))
    right_mask = apply_transform(ones, np.linalg.inv(H2))

    if display:
        log("display rectified images, with epipolar lines of select keypoints...")
        display_epipolar_lines(left_rect, right, F, H1, kps)

    return extractions | {
        "F": F,
        "H1": H1,
        "H2": H2,
        "left_rect": left_rect,
        "right_rect": right_rect,
        "left_mask": left_mask,
        "right_mask": right_mask,
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


def scanline(left, right, *, ltr=True, cost_f="SSD", block_size=11):
    cf = get_cost_f(cost_f)

    H, W = left.shape
    bs = block_size
    dmax = W // 5

    pad = block_size // 2
    left = np.pad(left, ((pad, pad), (pad, pad)), mode="constant")
    right = np.pad(right, ((pad, pad), (pad, pad)), mode="constant")

    left_patches = sliding_window_view(left, (bs, bs))
    right_patches = sliding_window_view(right, (bs, bs))

    cost_vol = np.full((H, W, dmax), np.inf)

    for d in trange(1, dmax, desc=f"{cost_f} - {bs}"):
        if ltr:
            rp = right_patches[:, d:]
            lp = left_patches[:, : -d or None]
            cost_vol[:, : -d or None, d] = cf(lp, rp)
        else:
            lp = left_patches[:, d:]
            rp = right_patches[:, : -d or None]
            cost_vol[:, d:, d] = cf(lp, rp)

    result = cost_vol.argmin(axis=2)
    return result if ltr else -result


def cross_check(left, right, *, left_mask=True, right_mask=True, cost_f='SSD', block_size=11):
    ltr = scanline(left, right, ltr=True, cost_f=cost_f, block_size=block_size)
    rtl = scanline(left, right, ltr=False, cost_f=cost_f, block_size=block_size)

    H, W = left.shape
    rows = np.arange(H)[:, None]

    xl = np.arange(W)
    xr = xl + ltr
    assert np.all(xr < W)

    left_match_map = np.isclose(-rtl[rows, xr], ltr, atol=3) & left_mask

    xr = np.arange(W)
    xl = xr + rtl
    assert np.all(xl >= 0)

    right_match_map = np.isclose(-ltr[rows, xl], rtl, atol=3) & right_mask

    return {
        'match_map': left_match_map,
        'left_match_map': left_match_map,
        'right_match_map': right_match_map,
        'ltr': ltr,
        'rtl': rtl,
    }


def get_cmap(type: str):
    if type == 'CC':
        cmap = plt.get_cmap('winter')
        cmap.set_bad('black')
    elif 'mask' in type.lower():
        cmap = None
    else:
        cmap = plt.get_cmap('cividis')
    return cmap


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
        cmap = get_cmap(type)
        for ax, (bs, dmap) in zip(axs, dmaps_bs.items()):
            if 'mask' in type.lower():
                dmap, match = dmap
                match_str = f", match: {match:1%}"
                dmin, dmax = False, True
            else:
                match_str = ''
                dmin, dmax = float(np.nanmin(dmap)), float(np.nanmax(dmap))

            im = ax.imshow(dmap, cmap=cmap)
            ax.set_title(f"{type} (bs: {bs}{match_str})")
            ax.axis("off")

            cbar = fig.colorbar(
                im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04
            )
            cbar.set_label("Disparity (px)", rotation=270, labelpad=15)
            cbar.set_ticks([dmin, dmax])
            cbar.set_ticklabels([dmin, dmax])

    plt.suptitle("Disparity Maps")
    plt.show()


def calculate_dmaps(
    left,
    right,
    *,
    scale=1 / 4,
    sad=False,
    ssd=True,  # faster, but visually equal
    ncc=False,
    cc=False,
    cc_cost_f=('SSD',),
    block_sizes=(5, 9, 15),
    logging=True,
    display=False,
    display_all=False,
):
    log = logger(logging)

    rect_res = rectify(left, right, logging=logging, display=display_all)
    left_rect, right_rect = rect_res["left_rect"], rect_res["right_rect"]

    left_rect_scal, right_rect_scal = rescale(left_rect, scale), rescale(
        right_rect, scale
    )

    disp_maps = defaultdict(dict)
    if sad:
        log("calculate SAD disparity maps.")
        for bs in block_sizes:
            disp_maps["SAD"][bs] = scanline(
                left_rect_scal, right_rect_scal, cost_f="SAD", block_size=bs
            )
        log("SAD disparity maps calculated.")
    if ssd:
        log("calculate SSD disparity maps.")
        for bs in block_sizes:
            disp_maps["SSD"][bs] = scanline(
                left_rect_scal, right_rect_scal, cost_f="SSD", block_size=bs
            )
        log("SSD disparity maps calculated.")
    if ncc:
        log("calculate NCC disparity maps.")
        for bs in block_sizes:
            disp_maps["NCC"][bs] = scanline(
                left_rect_scal, right_rect_scal, cost_f="NCC", block_size=bs
            )
        log("NCC disparity maps calculated.")
    if cc:
        log("calculate cross-check match maps.")
        left_mask = rescale(rect_res["left_mask"], scale)
        right_mask = rescale(rect_res["right_mask"], scale)
        for bs in block_sizes:
            for cost_f in cc_cost_f:
                result = cross_check(
                    left_rect_scal, right_rect_scal,
                    left_mask=left_mask, right_mask=right_mask,
                    cost_f=cost_f, block_size=bs)
                dmap = result['ltr'].astype(float)
                mmap = result['match_map']
                dmap[~mmap] = np.nan
                disp_maps[f"CC ({cost_f})"][bs] = dmap
                disp_maps[f"CC Match Mask ({cost_f})"][bs] = mmap, mmap.sum() / left_mask.sum()
        log("cross-check match maps calculated.")

    if display or display_all:
        log("display disparity maps...")
        plot_dmaps(disp_maps)

    return rect_res | {"disp_maps": disp_maps}
