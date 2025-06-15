import os

import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from skimage.feature import SIFT, match_descriptors
from skimage.transform import estimate_transform, warp
from sklearn.linear_model import RANSACRegressor

resource_folder = './resources/tiff'
image_size = 6


def read_images(path):
    return [iio.imread(os.path.join(path, file)) for file in os.listdir(path)]


images = read_images(resource_folder)


def get_kp_extractor(image):
    feat_detector = SIFT()  # each image needs it's own instance of SIFT anyway
    feat_detector.detect_and_extract(image)
    return feat_detector


def match(descriptors1, descriptors2):
    """
    :return: numpy array of shape (n, 2) of matched kp indices
    """
    return match_descriptors(descriptors1, descriptors2, cross_check=True, max_ratio=.8)


def rm_outliers(matched_kps):
    """
    :return: bool mask of inliers
    """
    model = RANSACRegressor(random_state=42)
    model.fit(matched_kps[:, 0], matched_kps[:, 1])
    return model.inlier_mask_


def calc_transform(matched_points):
    """
    :param matched_points:
    :return: inverse transform matrix
    """
    return estimate_transform('affine', src=matched_points[:, 0], dst=matched_points[:, 1]).params


apply_transform = warp


def display_transform(src_img, dest_img, aligned_dest_img):
    fig, axes = plt.subplots(ncols=2, figsize=(16, 8))

    fig.suptitle('KP Matched Transforms')

    axes[0].grid(False)
    axes[0].set_title('Original')
    axes[0].imshow(dest_img, cmap='gray')
    axes[0].imshow(src_img, cmap='gray', alpha=0.5)

    axes[1].grid(False)
    axes[1].set_title('Aligned')
    axes[1].imshow(src_img, cmap='gray')
    axes[1].imshow(aligned_dest_img, cmap='gray', alpha=0.5)

    plt.show()


def display_matches(image1, image2, filtered_matched_kp, k=100):
    k = min(k, len(filtered_matched_kp))
    fig, axes = plt.subplots(ncols=2, figsize=(16, 8))

    points1 = filtered_matched_kp[:, 0][:k]
    points2 = filtered_matched_kp[:, 1][:k]

    cmap = plt.get_cmap('tab10')
    c = [cmap(i) for i in np.linspace(0, 1, k)]

    fig.suptitle('Matching KPs')

    axes[0].grid(False)
    axes[0].imshow(image1, cmap='gray')
    axes[0].scatter(points1[:, 0], points1[:, 1], marker='x', c=c)

    axes[1].grid(False)
    axes[1].imshow(image2, cmap='gray')
    axes[1].scatter(points2[:, 0], points2[:, 1], marker='x', c=c)

    plt.show()


def extract_kps_and_calc_transform(image1, image2, max_kps=10000):
    ext1, ext2 = get_kp_extractor(image1), get_kp_extractor(image2)
    descriptors = ext1.descriptors[:max_kps], ext2.descriptors[:max_kps]
    kps = ext1.keypoints[:max_kps], ext2.keypoints[:max_kps]
    matches = match(*descriptors)
    matched_kps = np.concat((kps[0][:, ::-1][matches[:, 0]][:, None],
                             kps[1][:, ::-1][matches[:, 1]][:, None]),
                            axis=1)
    inlier_mask = rm_outliers(matched_kps)
    filtered_matched_kps = matched_kps[inlier_mask]
    return calc_transform(filtered_matched_kps)
