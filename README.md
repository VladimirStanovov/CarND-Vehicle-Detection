**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/CarImageHog.jpg
[image2]: ./output_images/NonCarImageHog.jpg
[image3]: ./output_images/SearchArea.jpg
[image4]: ./output_images/test1_boxes.jpg
[image5]: ./output_images/test1_final.jpg
[image6]: ./output_images/test1_heatmap.jpg
[image7]: ./output_images/test2_boxes.jpg
[image8]: ./output_images/test2_final.jpg
[image9]: ./output_images/test2_heatmap.jpg
[image10]: ./output_images/test3_boxes.jpg
[image11]: ./output_images/test3_final.jpg
[image12]: ./output_images/test3_heatmap.jpg
[image13]: ./output_images/test4_boxes.jpg
[image14]: ./output_images/test4_final.jpg
[image15]: ./output_images/test4_heatmap.jpg
[image16]: ./output_images/test5_boxes.jpg
[image17]: ./output_images/test5_final.jpg
[image18]: ./output_images/test5_heatmap.jpg
[image19]: ./output_images/test6_boxes.jpg
[image20]: ./output_images/test6_final.jpg
[image21]: ./output_images/test6_heatmap.jpg
[image22]: ./output_images/CarImage.jpg
[image23]: ./output_images/NonCarImage.jpg
[video1]: ./project_video.mp4

---
###Writeup / README

###Histogram of Oriented Gradients (HOG)

####1. HOG features extraction from the training images.

The code for this step is contained in the second code cell of the IPython notebook Project-5.ipynb. It also contains spatial features extraction, color histogram calculation and box drawing functions. 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image22]
![alt text][image23]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image1]
![alt text][image2]


####2. Setting the final choice of HOG parameters.

I tried various combinations of parameters, for color space types, best choises appeared to be YUV and YCrCb, I ended up using YCrCb as it delivered better classification quality. I've also tried to change the number of pixels per cell, which directly influences the amount of features in the feature vector, but `pixels_per_cell=(8, 8)` looks like a good tradeof between speed and quality. I also used all three HOG channels, and although the Y channel is most informative, the Cr and Cb still contain important color information. Changing the spatial features size, as well as number histogram bins does not change classification quality significantly. The resulting feature vector length is 6108.

####3. Training a classifier using selected features.

I trained a nonlinear SVM with rbf kernel, as linear SVM returned too much false positives on test images and videos. Other types of SVM kernels (polynomial, sigmoid, etc.) have similar quality to linear SVM, however take much more time to train and apply. For a linear SVM the typical test sample quality is around 0.985-0.987, while SVM with rbf kernel can achieve 0.994-0.996, which is a significant difference. However, the rbf kernel takes much more time to compute.

###Sliding Window Search

####1. Implementing a sliding window.

First of all, to implement sliding window, I used several scales: 1.4, 1.5 and 1.6 to get more robust detection in case of too many false positives or cars being too far or too close. Next, in the find_cars function in cell 10, I cut the are of interest, i.e. from 380 to 572 on vertical axis. This step is important for computation time reduction, we don't want to look for cars on the sky (usually there are none of them over there). Next, I extracted HOG features for the whole area of intetest, to be able to reuse these values during window sliding. The sliding itself had the overlap equal to 2 HOG blocks. using 1 HOG block for every step is also possible, but it results in increase of computation time. The overlap of 1 looked like a good option for me when I tried to use linear SVM, i.e. it resulted in more detections of the same vehicle, but too many false positives did not make it possible to get a good soultion even with large thresholds on heatmaps.

![alt text][image3]

####2. Some examples of test images to demonstrate how the pipeline is working.

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
![alt text][image7]
![alt text][image10]
![alt text][image13]
![alt text][image16]
![alt text][image19]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

