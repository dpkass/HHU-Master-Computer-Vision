import os
import pickle
import operator
from itertools import accumulate

import napari
import numpy as np
import imageio.v3 as iio
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from skimage.feature import SIFT, match_descriptors
from skimage.transform import estimate_transform, warp
from sklearn.linear_model import RANSACRegressor

resource_folder = './resources/tiff'
image_size = 6


def logger(on=True):
    return print if on else lambda *a, **kw: None


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


def extract_kps_and_calc_transform(image1, image2, *, max_kps=10000, logging=True, display=False):
    log = logger(logging)

    log('start kp extraction.')
    ext1, ext2 = get_kp_extractor(image1), get_kp_extractor(image2)
    log(f'extraction successful. image1: {len(ext1.descriptors)} kps | image2: {len(ext2.descriptors)} kps.')

    if len(ext1.descriptors) > max_kps or len(ext2.descriptors) > max_kps:
        log(f'too many kps. reducing to {max_kps} each.')
    descriptors = ext1.descriptors[:max_kps], ext2.descriptors[:max_kps]
    kps = ext1.keypoints[:max_kps], ext2.keypoints[:max_kps]

    log('start matching.')
    matches = match(*descriptors)
    log(f'matching successful. matched {len(matches)} kps.')

    matched_kps = np.concat((kps[0][:, ::-1][matches[:, 0]][:, None],
                             kps[1][:, ::-1][matches[:, 1]][:, None]),
                            axis=1)
    log('start outlier removal.')
    inlier_mask = rm_outliers(matched_kps)
    filtered_matched_kps = matched_kps[inlier_mask]
    log(f'outlier removal successful. dropped: {len(matched_kps) - len(filtered_matched_kps)} kps | remaining: {len(filtered_matched_kps)} kps.')

    log('calculate transform.')
    transform = calc_transform(filtered_matched_kps)
    log('transform calculated.')

    log('apply transform.')
    aligned_image2 = apply_transform(image2, transform)
    log('transform applied.')

    if display:
        log('display matches and transform...')
        display_matches(image1, image2, filtered_matched_kps)
        display_transform(image1, image2, aligned_image2)

    return {
        'keypoints': filtered_matched_kps,
        'transform': transform,
        'aligned_image': aligned_image2,
    }


def extract_kps_and_calc_transforms(images, max_kps=10000, logging=False, view=False):
    pbar = tqdm(zip(images, images[1:]), total=len(images) - 1, desc="Calculating Transforms")
    return [np.eye(3)] + [extract_kps_and_calc_transform(i1, i2, max_kps, logging, view)
                          for i1, i2 in pbar]


def align_images(images, transforms):
    stacked_transforms = list(accumulate(transforms, operator.matmul))
    aligned_images = [apply_transform(img, t) for img, t in zip(images, stacked_transforms)]
    return {
        'stacked_transforms': stacked_transforms,
        'aligned_images': aligned_images
    }


def pad_and_stack(aligned_images):
    """
    Pad images to the same size and stack them into a single array.

    Constant pad with the max value of the image, because alignment creates artifacts.
    """
    max_val = max(i.max() for i in images)
    max_h, max_w = max(i.shape[0] for i in aligned_images), max(i.shape[1] for i in aligned_images)

    return np.stack([np.pad(img, ((0, max_h - img.shape[0]), (0, max_w - img.shape[1])),
                            constant_values=max_val) for img in aligned_images])


def remove_alignment_artifacts(volume, threshold=0.95):
    volume = volume.copy()
    volume[volume > threshold] = 0
    volume /= volume.max()
    return volume


def view_volume(volume, colormap='bop purple'):
    napari.view_image(volume, colormap=colormap, rendering='mip', ndisplay=3, scale=(5, 1, 1))
    napari.run()


def transform_to_volume(images, *, save_path=None, view=False):
    """
    Transform a sequence of 2D images into a 3D volume by aligning and stacking them.
    
    This function performs the following steps:
    1. Extracts keypoints and calculates transforms between consecutive images
    2. Aligns all images using the calculated transforms
    3. Stacks aligned images into a 3D volume
    4. Removes alignment artifacts
    5. Optionally displays the resulting volume
    6. Optionally saves the transforms and volume to a file
    
    :param images: List of 2D numpy arrays representing the sequence of images to be transformed
    :param save_path: Optional path to save the resulting transforms and volume as a pickle file
    :return: None
    """
    transforms = extract_kps_and_calc_transforms(images)
    stacked_transforms, aligned_images = align_images(images, transforms)
    volume = pad_and_stack(aligned_images)
    volume = remove_alignment_artifacts(volume)
    if view: view_volume(volume)

    result = {
        'transforms': stacked_transforms,
        'volume': volume
    }

    if save_path:
        with open(save_path, 'wb') as file:
            pickle.dump(result, file)

    return result
