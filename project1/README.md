# Project 1: 3D reconstruction from 2D brain scans

In this project, we would like you to investigate image registration.
For this purpose, we will give you a batch of unregistered brain images from our research group.
Your task is to apply the methods from the lecture and transform these images onto each other to obtain a 3D representation of the original brain.
The exercises in the coming weeks will guide you through the steps to accomplish this task.

![A stack of images](./supplementary/stacking.png)

Submit your results as described in the [README](https://git.hhu.de/2025-ss-computervision-exercises/2025-ss-computervision-readme/-/blob/main/README.md?ref_type=heads).

## Data

The image data for this project can be found here: [sciebo](https://fz-juelich.sciebo.de/s/oyMbloDwucGLhNu).
**Note:** Please make sure to not submit the input data with your solution.

## Exercise 1

**Due date:** 29.04.2025, 23:59

**Learning goals:** reading and annotating images, estimating 2D transformations, applying 2D transformations

In this exercise, the goal is to familiarize yourself with the data, and manually estimate the transformations between an example pair of images.
After inspecting the images, you will manually select corresponding points in two images and estimate transformation matrices from the selected points.

1. Read some of the provided images (e.g., using `imageio`) and visualize them (e.g., using `matplotlib` or `napari`).
2. Print statistics of the images read, including their type, shapes, and intensity statistics (e.g., min, max, mean, median).
3. Select a pair of adjacent images. In these images, identify pairs of corresponding points in both images (**hint:** Which points are easiest to identify consistently in both images?). There are several ways to select the points. We recommend familiarizing yourself with [napari](https://napari.org/dev/index.html), which can be used to visualize and annotate images. Alternatively, plotting tools like `matplotlib` typically support [interactive figures](https://matplotlib.org/stable/users/explain/figure/interactive.html), which allow you to extract the coordinates of the pixels under your mouse pointer. **Hint:** Save identified pairs so that you can reuse them later.
4. Using the corresponding points, estimate linear transformations (rigid and affine) to align the images.
5. Align the images using the computed transformations and overlay them (e.g., using the `alpha` parameter in `matplotlib`).

# Exercise 2

**Due date:** 06.05.2025, 23:59

**Learning goals:** image filtering; downsampling, histogram equalization

In this exercise, you will learn different techniques that are useful for many image analysis projects, including the 3D reconstruction of brain scans.
The goal of this exercise is to implement several image operations yourself.
Using basic `numpy` operations is allowed (e.g., using `np.mean` on one sliding window), but more complex operations must be implemented by yourself (e.g., the logic for a `convolution`).

1. Using at least two images, implement a function that downsamples images using linear interpolation.
2. Using the same images, perform a histogram equalization. Compute the histogram equalization on the downsampled images, then apply it to the original images. In your solution, comment on the advantage of this approach over computing the histogram equalization on the original images only.
3. Implement and apply the following filters on the original and downsampled images, and compare the results 1) mean 2) median 3) Gaussian 4) Sobel filter.
4. Using your Gaussian filter implementation, extent your downsampling implementation to include anti-aliasing.

## Exercise 3

**Due date:** 13.05.2025, 23:59

**Learning goals:** find points of interest, compute key point descriptors, match key point descriptors

Having learnt how to estimate transformations from manually selected key points in exercise 1, you will now learn how to find and match points of interest automatically.

1. Given a pair of adjacent images, use feature detectors from `scikit-image` or `OpenCV` to detect key points in both images (e.g., [SIFT](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.SIFT)).
2. Match the identified points using the matching algorithm described in the lecture.
3. Remove outliers using `RANSAC` (you may use existing implementations).
4. Using your code from exercise 1, calculate transformations between the two images.
5. Apply transformations and inspect the results as in exercise 1.
6. Experiment with filters from exercise 2 to improve results or runtime performance.

## Project 1

**Due date:** 20.05.2025

**Learning goals:** put together what you learned

In this project, what you learned in the exercises comes together to tackle the 3D reconstruction of 2D brain scans.

1. Use your approach from exercise 3 to compute transformations between any pair of adjacent images.
2. Using one image (e.g., the first one, or the one in the center) as a reference, concatenate and apply the computed transformations to align all images to your reference image.
3. Stack the transformed images into a 3D volume and inspect the result using `napari`.
