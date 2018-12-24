# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[train_dist]: ./visualizations/train_dist.png "Train dist"
[val_dist]: ./visualizations/val_dist.png "Val dist"
[test_dist]: ./visualizations/test_dist.png "Test dist"
[sample_images]: ./visualizations/sample_images.png "Sample images"
[test_13]: ./test_images/13.png "Test Image 13"
[test_25]: ./test_images/25.png "Test Image 25"
[test_27]: ./test_images/27.png "Test Image 27"
[test_33]: ./test_images/33.png "Test Image 33"
[test_37]: ./test_images/37.png "Test Image 37"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Cribbin/self-driving-car-engineer-nanodegree-3/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used regular Python and numpy to calculate the summary

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3 (3 colour channels)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here are some exploratoy visualizations of the dataset:

Distribution of classes in the training dataset:
![train_dist][train_dist]

Distribution of classes in the validation dataset:
![val_dist][val_dist]

Distribution of classes in the test dataset:
![test_dist][test_dist]

The following is a random sampling of the training dataset, along with their associated labels:
![sample_images][sample_images]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


The only preprocessing step I performed was to normalize the images. This results in the images having mean zero and equal variance to allow for quicker accurate classification. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 					|
| Flattened				| Outputs 400        							|
| Fully connected		| Outputs 120        							|
| RELU					|												|
| Dropout				| Keep prob: 50% 								|
| Fully connected		| Outputs 84        							|
| RELU					|												|
| Dropout				| Keep prob: 50% 								|
| Softmax				| 43 classes        							|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I minimized mean cross entropy using the Adam Optimizer, with a learning rate of 0.0005. Dropout was set to 0.5, 25 epochs, and batch size of 128.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

I chose initially a LeNet architecture. I achieved a validation accuracy of around 87% with this. I looked at other architectures, and found the AlexNet architecture. One of the advances this architecture took advantage of was to use dropout. I added dropout to the two fully connected hidden layers, with increased my accuracy to over 90%. I then decreased the learning rate and increased the number of epochs, when I encountered the current vlaidation accuracy of 94.6%.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

The images I chose are from the following video:
[![Video](http://img.youtube.com/vi/2LXwr2bRNic/0.jpg)](http://www.youtube.com/watch?v=2LXwr2bRNic)

I chose this video for two main reasons:
* It will provide realistic examples of how signs might appear from a front-facing camera in a vehicle, taking into account factors like shadows, glare etc.
* I was worried that getting direct images online might result in me accidentally retrieving similar/identical images to the dataset used for training.

##### 33 - Turn right ahead
![test_33][test_33]
* `16:13` in the video


##### 37 - Go straight or left
![test_37][test_37]
* `11:35` in the video

##### 25 - Road work
![test_25][test_25]
* `20:01` in the video
* I feel this is the most difficult image. It is at a very sharp angle, and a lot of the information is lost when scaling it to 32*32 (It's a roadworkds sign). It also contains part of another sign in the image. Due to this, it should be a good test of how robust my classifier is.

##### 27 - Pedestrians
![test_27][test_27] 
* `02:00` in the video
* This one shares all three problems mentioned in the last image, but to a lesser degree.

##### 13 - Yield
![test_13][test_13] 
* `25:40` in the video
* I feel this might test the model as the color information for the yield sign appears to not appear fully in this frame, the border appears black isntead of red.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image					|     Prediction								| 
|:---------------------:|:---------------------------------------------:| 
| Turn right ahead		| Stop sign   									| 
| Go straight or left	| U-turn 										|
| Road work				| Yield											|
| Pedestrians			| Bumpy Road					 				|
| Yield					| Slippery Road									|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
This step wasn't performed.

