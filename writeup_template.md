# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image9]: ./German-Traffic-Signs/german-road-signs-animals.png "Traffic Sign - Animals"
[image10]: ./German-Traffic-Signs/german-road-signs-pedestrians.png "Traffic Sign - Pedestrains"
[image11]: ./German-Traffic-Signs/german-road-signs-slippery.png "Traffic Sign - Slippery Road"
[image12]: ./German-Traffic-Signs/german-road-signs-traffic.png "Traffic Sign - Traffic Signal"
[image13]: ./German-Traffic-Signs/germany-min-speed-sign.png "Traffic Sign - Min Speed"
[image14]: ./training-data-visualization.png "Training Data Visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/dinesh-i/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 X 32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the training data set. It is a line chart showing the count of training examples available for each of the classes

![alt text][image14]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Preprocessed the image with the following 2 steps:
- Normalized the data and converted the data to the range of (-1,1)
- Converted the image to Gray Scale


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray Scale image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16  				|
| Flatten					|			outputs 400									|
| Fully connected		| outputs 120        									|
| RELU					|												|
| Fully connected		| outputs 84        									|
| RELU					|												|
| Fully connected		| outputs 10        									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I chose Adam Optimizer with a learning rate of 0.0095 and epsilon of 0.01 with 25 EPOCHs. This resulted in a validation accuracy of 93%


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.


Trained the model with different optimizers and below are the results:

| Optimizer         		|     Learning Rate	|     Optimizer Specific Arguments		|Epoch| Validation Accuracy |
|:---------------------:|:---------:|:---------------------:|:---------------------:|:---------------------:|
|AdadeltaOptimizer|0.01||35|0.821 |
|  FtrlOptimizer|0.01||35|0.054|
|  GradientDescentOptimizer |0.01||35|0.824|
|  GradientDescentOptimizer |0.001||35|0.111|
|  MomentumOptimizer |0.001|momentum=0.9|35|0.829|
|  MomentumOptimizer |0.01|momentum=0.9|35|0.900|
|  MomentumOptimizer |0.01|momentum=1.5|35|0.007|
|  MomentumOptimizer |0.01|momentum=0.5|35|0.857|
|  AdamOptimizer |0.0065|epsilon=default|35|0.919|
|  AdamOptimizer |0.0065|epsilon=1.0|35|0.724|
|  AdamOptimizer |0.0065|epsilon=0.1|35|0.875|
|  AdamOptimizer |0.0065|epsilon=0.01|35|0.899|
|  AdamOptimizer |0.009|epsilon=0.01|35|0.943|
|  AdamOptimizer |0.0095|epsilon=0.01|25|0.93|

Based on the above test results I chose the last entry shown in the above table. This resulted in a validation accuracy of 93%, test accuracy of 92.1% and worked for 60% of the new images taken from the web.



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image12] ![alt text][image11] ![alt text][image10] 
![alt text][image9] ![alt text][image13]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Traffic Signals				| Traffic Signals											|
| Slippery Road			| Slippery Road      							|
| Pedestrain     			| General Caution 										|
| Animals      		| Animals   									| 
| 30 Min Speed Sign	      		| Yield					 				|



The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares favorably to the accuracy on the test set of 92.1%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is totally sure that this is a traffic signal (probability of 1.0). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Traffic signals   									| 
| 5.01383424e-09     				| General caution 										|
| 1.09423898e-19					| Road narrows on the right											|
| 7.95560535e-20	      			| Pedestrians			 				|
| 7.44806994e-20				    | Keep right     							|


For the second image, the model predicted the correct sign with relatively better probability(50.4%). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.504100740e-01         			| Slippery road			| 
| 4.95899320e-01|Dangerous curve to the left |
| 2.03150579e-31	|Speed limit (60km/h) |
| 1.04777395e-31	|Double curve |
| 3.22595352e-32| No passing for vehicles over 3.5 metric tons|


For the third image, the model didn't predict the correct sign. Although the second option was the correct sign, the probability is pretty less compared to the first option chosen by the model. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.990585268	| General caution	| 
| 9.41472314e-03	| Pedestrians	| 
| 1.87533348e-20	| Right-of-way at the next intersection	| 
| 2.93237782e-24	| Speed limit (30km/h)	| 
| 5.05530122e-25	| Roundabout mandatory	| 


For the fourth image, the model is totally sure that this is a 'wild animals crossing' signal (probability of 1.0). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|1	| Wild animals crossing	| 
|6.40190976e-18	| Road work	| 
|1.83175673e-20	| Double curve	| 
|	1.68070406e-26| Slippery road	| 
| 9.97941145e-33	| Beware of ice/snow	| 


For the fifth image, the model didn't predict the correct sign. The actual sign is not part of the top 5 probabilities. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|0.964999557	| Yield	| 
|3.32169235e-02	|Children crossing 	| 
|1.59830658e-03	| Right-of-way at the next intersection	| 
|1.84327742e-04	| Roundabout mandatory	| 
|8.65226923e-07	| Traffic signals	| 


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


