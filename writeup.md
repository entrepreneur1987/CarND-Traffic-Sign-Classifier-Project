# Traffic Sign Recognition

### Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:

Load the data set (see below for links to the project data set)
Explore, summarize and visualize the data set
Design, train and test a model architecture
Use the model to make predictions on new images
Analyze the softmax probabilities of the new images
Summarize the results with a written report
Rubric Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/entrepreneur1987/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set:

The size of training set is 34799
The size of the validation set is 4410
The size of test set is 4410
The shape of a traffic sign image is (32,32,3)
The number of unique classes/labels in the data set is 43

### Design and Test a Model Architecture

#### Data Pre-processing
I did some data augmentation by rotating images whose class has less examples to make the distribution more balanced.

I also converted the images to grayscale because same traffic sign class may contain images with different colors.

As a last step, I normalized the image data because it will make training faster.

#### Model architecture summary
My final model consisted of the following layers:

Layer   | Description
--- | ---
Input   |32x32x1 GrayScale image
Convolution |5x5 1x1 stride, VALID padding, outputs 28x28x16
RELU |   
Max pooling |2x2 stride, outputs 14x14x16
Convolution |5x5 1x1 stride, VALID padding, outputs 10x10x128
Max pooling |2x2 stride, outputs 5x5x128
Dropout with |keep_prod = 0.7
Fully connected |3200->800
RELU|
Fully connected |800->400
RELU|
Fully connected |400->120
RELU|
Fully connected |120->84
RELU|
Fully connected |84->43

To train the model, I used Adam optimizer, with BATCH_SIZE = 128, EPOCH = 30 and learning rate = 0.001

My modeled is based on LeNet, with addition of dropout and higher depth, which enhances the accuracy.

My final model results were:

training set accuracy of 0.999

validation set accuracy of 0.946

test set accuracy of 0.935

If an iterative approach was chosen:

#### Q & A
What was the first architecture that was tried and why was it chosen? LeNet, because it's simple
What were some problems with the initial architecture? Accuracy stuck at 0.92
How was the architecture adjusted and why was it adjusted? Added dropout layer, as well as increasing the depth of the convolutional layer

### Test on images from the web
Here are seven German traffic signs that I found on the web: 
Double curve, No entry, Children passing, Road work, speed 30, Stop and Yield

The speed 30 image might be difficult to classify because the number gets wrong sometimes like becoming speed 50/20.

Here are the results of the prediction:

Image |   Prediction
--- | ---
Double curve |Double curve/Road work(different runs)
No entry |No entry
Children passing |Children passing
Road work |Road work
Speed 30km/h |Speed 30km/h
Stop Sign |Stop sign
Yield   |Yield

The model was able to correctly guess 7 of the 7 traffic signs, which gives an accuracy of 80%-100%. This compares favorably to the accuracy on the test set of 93.5%



The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

The top five soft max probabilities for the first image were (predicted wrongly)

Doulbe curve:
Probability |Prediction
--- |---
.770        |Road work 
.215        |Doulbe curve 
.014        |Slipper road 
.0002       |Dangerous curve to the left 
.000058     |Wild animal crossing 


