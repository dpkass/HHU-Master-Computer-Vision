# Project 2: Scene Reconstruction

This project addresses scene reconstruction.
You will use data from the public dataset [2014 Stereo Dataset](https://vision.middlebury.edu/stereo/data/scenes2014/), which contains stereo images of different objects.
You will estimate disparity and depth maps from the images, apply graphical models to smooth the results, and perform surface extraction of the 3D scene.

Submit your results as described in [the README](https://git.hhu.de/2025-ss-computervision-exercises/2025-ss-computervision-readme/README.md).

## Exercise 1

**Due date:** 03.06.2025, 23:59

**Learning goals:** : F-matrix estimation

The goal of this exercise is to implement the F-matrix estimation to rectify two stereo images.

1. Take pairs of images with your camera/smartphone as basis for this task. Experiment with different kinds of images (e.g., different scenes, different lightning) to see how it affects the task. **Note:** You do not need to include the images with your submission, but please submit your notebook executed, so we can see the images plotted in the notebook.
2. Use what you learned about feature descriptors and key point matching to find correspondences between the two images. Plot the detected points and matches. **Tip:** Also think about outlier removal.
3. Use identified matches to estimate the fundamental matrix (F-matrix).
4. Apply the estimated F-matrix to rectify the images.

You may use existing implementations for key point detection and matching, but you should implement the matrix estimation yourself.

# Exercise 2

**Due date:** 10.06.2025, 23:59

**Learning goals:** compute scene depth using block matching

After rectifying images in the last exercise, the goal of this exercise will be to compute the scene depth.
You can use the images you took in the last exercise, or use some from the [2014 Stereo Datasets](https://vision.middlebury.edu/stereo/data/scenes2014/).

1. Display the images you selected for this task. Experiment with different pairs of images (e.g., different lightning).
2. Compute the disparity using the methods described in the lecture (scanline, cross-checking, resolution of ambiguous matches).
3. Use the computed disparity to estimate and display the depth.

For all steps, implement algorithms yourself.

**Tip:** If your implementation is too slow, you may downscale images to speed up computation.

## Exercise 3

**Due date:** 17.06.2025, 23:59

**Learning goals:** smooth depth maps using an MRF

In the last exercise, you computed depth from disparity maps.
To improve the quality of the depth estimation, you will now apply a Markov Random Field (MRF) to smooth the disparity maps.

1. Install a library of your choice for modelling the task as an MRF. We recommend [gco-wrapper](https://github.com/Borda/pyGCO).
2. Convert the task of smoothing the disparity map into a MRF problem. In particular, think about how labels can be defined for this task (tip: maximum disparity).,
3. Recompute the depth map using your corrected disparity maps and compare them to the previous result. Repeat the analysis for your selected images under different lightning and exposure.

## Project 2

**Due date:** 24.06.2025

**Learning goals:** scene surface construction

In this final exercise for this project, you will put together what you learned, and apply surface extraction to create scene surfaces from the estimated depth map.

1. Implement the marching squares algorithm to extract isosurfaces from images.
2. Apply your algorithm to your previously created depth maps.
3. Plot the computed surfaces on top of the original images.
