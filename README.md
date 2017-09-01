
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
[image3]: ./imagelinks/german_examples.png "germans"

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

There are two convolution layers and three fully connected layers, with the last being the logits.
In between layers 2 and 3, the image gets flattened. 

Layer 1 to 2 takes an input of 32x32x1, uses VALID padding, and outputs to a nuerons with dimensions 28x28x6. 
The convolution filter had shape 5x5x1x6 and the stride was 1 and max pooling was performed.

Layer 2 to 3 takes the input of 28x28x6, uses VALID padding, and outputs to nuerons with dimensions 10x10x16.
The covolution filter had shape 1x2x2x1 and the stride was 1 and max pooling was performed.

Layer 3 to 4 takes the input of 10x10x16, uses VALID padding, and outputs to nuerons with dimensions 5x5x16.
The covolution filter had shape 1x2x2x1 and the stride was 1 and max pooling was performed.

Next, the image was flattnened and the layers were fully connected.
Layer 3 to 4 The input was 5x5x16 which was output to 400 nuerons.
Layer 4 to 5 takes 400 nuerons and connects to 120 nuerons.
Layer 5 to output, the last layer connects the 84 nuerons to the 43 output classes.


The activation function used in each nueron was the Rectified Linear Unit.

####3. Trained Model Hyperparameters

The training was done on a NVIDIA Quadro M1000M graphics card. The solution converged with 50 Epochs, 
a batch size of 120 and a learning rate of 0.00097

####4. Approach Taken

As mentioned, the exact same LeNet network was used as in the lab example.
The LeNet architecture ended up performing the classification well, with a training accuracy of 100% and a validation accuracy of 94%. I first used a learning rate of 0.001 with only 20 EPOCHS. I saw that the training set accuracy was not hitting 100%, which it should. So by slowing down the rate of learning and increasing the time to learn, the training set accuracy would reach 100%.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 93.4%
* test set accuracy of 92%

###Test a Model on New Images

####1. German Traffic Signs

6 german traffic signs were acquired from off the web. 
The signs contained the same dimensionality as the train images.

![alt text][image3]

####2. Model's Prediction

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit (30km/h)  | Speed Limit (30km/h)  						| 
| Bumpy Road   			| Bumpy Road							|
| Ahead Only			| Ahead Only									|
| No Vehicles	    	| Speed Limit (50 km/h)  				 				|
| Go straight or Left	| Go straight or Left							|
| General Caution		| General Caution    							|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%.
The test set accuracy was 92%. The granularity of having only 6 traffic signs is 16%, so perhaps by having more traffic signs from the web, the test set accuracy of 92% would be represented. 

####3. Softmax Probabilities for German Signs
Below are the predictions.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit (30km/h)  | Correct, 100% 									| 
| Bumpy Road   			| Correct, 99%								|
| Ahead Only			| Correct, 99%									|
| No Vehicles	    	| Incorrect, 98%			 					|
| Go straight or Left	| Correct, 100%									|
| General Caution		| Correct, 100%    								|

The third traffic sign were the culprits, i.e. got classified incorrectly.
It is interesting to note that the probability was only 98% confident, compared to nearly 99% or 100% in all other 4 images.
This suggests that the network knew it was struggling with the image.
In a real world application, we could set a lower limit on predictions confidence.
In the case of the self driving car, we could say the car will only take action when it is >99% confident. 
Otherwise, notify the driver and trust the driver's input.