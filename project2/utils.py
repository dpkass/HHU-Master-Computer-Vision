import matplotlib.pyplot as plt
import numpy as np


def compute_two_boundary(H, W, a, b, c):
    x0, x1 = np.zeros_like(a), np.full_like(a, W)
    y0 = (-(a * x0 + c) / b)
    y1 = (-(a * x1 + c) / b)
    return np.stack([x0, x1]).clip(0, W - 1), np.stack([y0, y1]).clip(0, H - 1)


def display_epipolar_lines(img1_rect, img2_original, F, H1, kps, *, sample=True, k=5):
    if sample:
        idxs = np.random.choice(len(kps), size=k, replace=False)
        lpts, rpts = kps[idxs, 0], kps[idxs, 1]
    else:
        lpts, rpts = kps[:, 0], kps[:, 1]

    rpts_h = np.hstack([rpts, np.ones((len(rpts), 1))])  # shape (k, 3)

    F_rect = np.linalg.inv(H1).T @ F
    lines = (F_rect @ rpts_h.T).T  # shape (k, 3)

    x, y = compute_two_boundary(*img1_rect.shape, lines[:, 0], lines[:, 1], lines[:, 2])

    fig, axes = plt.subplots(ncols=2, figsize=(16, 5))

    cmap = plt.get_cmap('viridis')
    c = [cmap(i) for i in np.linspace(0, 1, k)]

    axes[0].imshow(img1_rect, cmap='gray')
    for i, col in enumerate(c):
        axes[0].plot(x[:, i], y[:, i], color=col)
    axes[0].axis('off')
    axes[0].set_title('Rectified Left Image with Epipolar Lines')

    axes[1].imshow(img2_original, cmap='gray')
    axes[1].scatter(rpts[:, 0], rpts[:, 1], marker='x', c=c)
    axes[1].axis('off')
    axes[1].set_title('Original Right Image with Keypoints')

    fig.suptitle("Rectified Images")
    plt.show()
