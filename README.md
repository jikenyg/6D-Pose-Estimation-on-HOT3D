![network](network.jpg)
# Contributions
(1) To address the severe distortion and viewpoint ambiguity caused by the large field of view in fisheye images, we propose a fisheye undistortion method and release an undistorted version of the HOT3D dataset.<br>
(2) For the problem of grayscale feature extraction, we design an optimized input strategy specifically tailored to grayscale images.<br>
(3) To enhance the discriminability of feature representations without incurring a significant increase in computational cost, we introduce an efficient multi-scale attention module.<br>
(4) Considering that the HOT3D dataset consists of grayscale fisheye distorted images, we conduct 6D object pose estimation on this dataset, integrating all the above components and improving pose estimation accuracy.<br>

# Distortion-Free Dataset Acquisition

To download the HOT3D dataset, please click the following link: [HOT3D Dataset](https://huggingface.co/datasets/bop-benchmark/hot3d/tree/main)<br>

For the repository to convert it into BOP format, please visit: [this link](https://github.com/facebookresearch/hot3d/tree/main/hot3d)<br>

Finally, please run the distortion-free script in this project:undistort_script.py


# Acknowledgements

Some code is based on MRC-NET. We thank the authors for their contributions!


