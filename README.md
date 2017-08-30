
**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./imagelinks/sign_color1.png "signcolor"
[image2]: ./imagelinks/sign_gray1.png "signgray"

[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---
###Data Set Summary & Exploration

####1. Provide a basic summary of the data set.

The number of training examples is roughly 3 times the size of the testing examples. 
Each image is has dimension 32x32x3, but I am planning on converting to gray scale.
There are 43 different traffic signs that the network will attempt to classify.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

The 32x32 image shows a good amount of pixelation.
While this low resolution will allow the images to back and forward through the network, 
there is less spatial relationship's that are possible in each image the network can learn.

![alt text][image1]

###Design and Test a Model Architecture

#### Basic Data Set Summary

The training images were all stripped of their 3 channel color dimension and changed to grayscale..
In addition, the training images were normalized, so that the grayscale values ranged from 0 to 1.
Grayscale and normalization give the network less relationships to learn. I chose not to alter or augment the data set, because the classifier converged with nearly a 95% percent accuracy through just the training set.
Had I wanted higher accuracy, I could rotate the image, blur the image, sharpen the image, etc.

![alt text][image2]


####2.Model Architecture

The LeNet architecture was used almost exactly the same as the lab exercise.
The activation function is the Rectified Linear Unit.
VALID padding is used.
There are two convolution layers and three fully connected layers, with the last being the logits.
In between layers 2 and 3, the image gets flattened. 

####3. Trained Model Hyperparameters

The training was done on a NVIDIA Quadro M1000M graphics card. The solution converged with 200 Epochs, 
a batch size of 156 and a learning rate of 0.00097

####4. Approach Taken

As mentioned, the exact same LeNet network was used as in the lab example.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 95% 
* test set accuracy of 95%

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

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
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


