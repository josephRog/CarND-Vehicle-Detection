## Vehicle Detection Project
### Joseph Rogers

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./report_images/base.png
[image2]: ./report_images/dataset_examples.png
[image3]: ./report_images/hog_example.png
[image4]: ./report_images/normalization.png
[image5]: ./report_images/search_area.png
[image6]: ./report_images/detections.png
[image7]: ./report_images/heat.png
[image8]: ./report_images/final.png 
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

Here is a link to my full [github repository.](https://github.com/josephRog/CarND-Vehicle-Detection)
A link to my final output video can be found [here](./output_video.mp4).

All code used to process the videos is contained within the `vehicle_detection.ipynb` jupyter notebook file.

Much of the code for this project has been borrowed and adapted from the Udacity Self-Driving Car Nanodegree Vehicle Detection and Tracking lectures.


### 1. Import Labeled Dataset for Training Linear SVM Classifier

The first thing that had to be done was to import the labeled training data that would be used to train the classifier. A series of 8782 car images and 8968 NOT car images were loaded. This images were borrow from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html) and the [KITTI vision benchmark suite.](http://www.cvlibs.net/datasets/kitti/) Loading of the dataset was performed in section 2 of the jupyter notebook.

A random example of each image type is shown below:

![alt text][image2]


### 2. Histogram of Oriented Gradients (HOG)

The next step was extract HOG features from all of the images in the dataset. HOG features were extracted from the pictures of car and NOT car images to be used with training the SVM classifier. The function `get_hog_features()`, borrowed from the udacity lectures, is viewable in section 3 of the code.

Example of HOG extraction on an image:

![alt text][image3]

There were several different parameters controlling how to perform the HOG extraction. They were `color_space`, `orient`, `pix_per_cell`, `cell_per_block`, and `hog_channel`. I tuned these values through experimentation. I changed them both off the reported accuracy of the SVM classifier and the qualitative accuracy they yielded when detecting cars in the video stream. In the final version of the code I used the values listed below.

| Parameter        | Value  | 
|:-------------:|:-------------:| 
| `color_space`      | 'YUV'       | 
| `orient`      | 10       | 
| `pix_per_cell`     | 16         |
| `cell_per_block`      | 2        |
| `hog_channel`     | 'ALL'      |


### 3. Feature Extraction and Training the SVM Classifier

The classifier I chose to train was a linear SVM that looked at features extracted from the car and NOT car dataset. In addition to HOG features, it also operated on spatial features from the images, and regular color histogram features.

All three types of features were extracted using the `extract_features()` function presented in section 6 of the code. Extracted features were normalized to have a 0 mean. This was done to assist accurate training of the classifier.

An example of normalizing the features extracted from an image.

![alt text][image4]

All features were extracted using the `YUV` color space. This was found to be siginificatly faster that using other color spaces and yielded nice results when run on the video. A size of 16 was used for the number of color histogram bins and a spatial binning size of 16x16 was used for spatial features.

Total time to extract all features from the dataset took about 60 seconds. Once this was complete, training the classifier only took about 2.5 seconds. The accuracy was consistently reported to be above 0.99%.

### 4. Sliding Window Search

Once I had trained the classifier, I used it to sample some pieces out of image frames using a sliding window technique. Since cars are only likely to appear in certain areas of the image frame and at different relative sizes, I restricted my search to only be in certain places based on the sampling size of the window. I ended up doing four search passes on each image frame. I started with smaller samples and then moved to larger ones. As the sample size increased I widened the search area and moved it down.

This image shows the relative search areas for each of the four search passes.

![alt text][image5]

Using this technique made searching MUCH faster, due to the restricted area. It was not perfectly accurate however as the white car is slightly outside the search area. To compensate for this I adjusted the right sides of the search areas out a bit in the final processing pipeline.

### 5. Vehicle Detections

Once the search area were properly defined, I extracted the features of each image frame and then searched them using the classifer to determine whether or not a vehicle was present. This was done by the `find_cars()` function, available in section 13 of the notebook code.

Using the sliding window search with the trained linear SVM classifier produces the following result on the test image.

![alt text][image6]

This generally look okay, however it can be seen that there is a spurious detection in the middle of the road and that there are multiple detections on each car.

### 6. Filter False Positives and Multiple Detections

In order to filter for false positives and multiple or overlapping vehicle detections, I implemented the use of a heatmap to identify and isolate only the most important areas of interest. This was done in section 14 by the functions `add_heat()`, `apply_threshold()`, and `draw_labeled_bboxes()`.

![alt text][image7]

The two images above demonstrate the raw heat output when mapping all the detections of the image frame. Once a threshold is applied dropping lower values down to black, only the right hand image remains. When this data is used to redraw detections back onto the image frame, the following final image is produced.

![alt text][image8]


---

## Video Implementation and Smoothing
A link to my final output video can be found [here](./output_video.mp4). 

In order to try and smooth out the drawing of boxes in between frames, I impleted a new class `Vehicle`. The `Vehicle` class keeps track of interframe data so that vehicle detections from previous frames still have influence over where boxes are drawn on the current frame.

The class keeps track of the currect frame count and all of the detections during some previous (e.g. 5) number of frames. When it comes time in the pipeline to do the heatmaps, all of the previous detections are added to the currect frames detections. As a result of many more detections, a much higher threshold value is needed to effectivly isolate the vehicles. However, the affect of this is that many more false detections get filtered out even if they also had few detection on them. The use of this class produces a much smoother output. The code for the `Vehicle` class can be found in section 16.


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Though the SVM classifier was able to consistently trained to a high level of accuracy (e.g. > 99%), I found that it often performed with mixed results on the actual video stream. This seem indicative to me of overfitting to the dataset and not generalizing well to the real world. I was able to get it working well enough eventually, but this required somewhat meticulous tuning of parameters. It seems unlikely to me that it would work very well on a different road, on a different day with other cars unless the parameters were re-tuned again. To try and fix this problem I would try to shuffle my data better and maybe augment it with some warped, scalled, etc.. images to try and reduce its overfitting. I might also try using a different method than SVMs.

To increase the speed of the classifier, I restricted the search field pretty severely. This may be a problem on a wider highway with more lanes or when on certain kinds of hills where vehicles might appear higher up in the field of view. It seems possible that this could be easily addressed by dynamically adjusting the search area for cars to follow the areas as detected by the lane finder. This would allow the search space to still be relatively tight, but to move around the field of view yielding good performance, and not missing vehicles that should be detected.

