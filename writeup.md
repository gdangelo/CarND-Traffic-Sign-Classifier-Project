**Build a Traffic Sign Recognition Project - Grégory D'Angelo**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[dataset_distribution]: ./plots/training_visualization.png "Dataset distribution"
[training_images]: ./plots/random_examples.png "Training Images"
[training_preprocessing]: ./plots/training_preprocessing.png "Preprocessing visualization"
[learning_curves]: ./plots/learning_curves.png "Learning curves"
[new_traffic_signs]: ./plots/new_images.png "New Traffic Signs"
[top5_softmax_prob]: ./plots/top5_softmax_prob.png "Top 5 Softmax Probabilities"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation. The source code could be found [here](https://github.com/gdangelo/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

---
###Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes/labels in the data set is **43**

Here is an exploratory visualization of the training dataset (validation and testing dataset distribution could be found [here](https://github.com/gdangelo/CarND-Traffic-Sign-Classifier-Project/tree/master/plots)). It is a colored bar chart showing how the data is distributed across the different labels/sign names.

![Dataset distribution][dataset_distribution]

Finally, here is a visualization of 10 random images from the training dataset for each 43 classes. Due to weather conditions, time of the day, lighting, and image orientation we can notice big differences in appearance between each image.

![Training images][training_images]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

After some experiments and research, I decided to apply 3 preprocessing techniques to the image data in the following order: histogram equalization, grayscaling, and normalization.

Here is an example of a traffic sign image before and after each preprocessing steps.

![Preprocessing steps][training_preprocessing]

First, I applied histogram equalization because it improved feature extraction even further. 

Then I convert the images into grayscale, because color channels didn't seem to improve the model, but it tends to be just more complex at the end and slows down the training.

Finally, normalization was applied to scale pixel values from [0, 255] to [0, 1] for mathematical reasons. It is required for optimizer algorithm that I'll use to train the model.

I decided not to generate additional data for now because the accuracy obtained on the test dataset is quite statifying (96%). As a future improvement, I could manually look at the individual images for which the model makes a wrong judgement. So that, I then can augment the dataset accordingly.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I decided to use a deep neural network classifier as a model, which was inspired by LeNet-5. It is composed of 7 layers: 4 convolutional layers for feature extraction and 3 fully connected layer as a classifier.

Here are the details of the architecture:

| Layer           | Description                                 |
|:---------------:|:-------------------------------------------:|
| Input           | 32x32x1 image (after preprocessing steps    | 
| Convolution 3x3 | 1x1 stride, valid padding, outputs 30x30x32 |
| RELU            |                                             |
| Max pooling     | 2x2 stride, outputs 15x15x32                |
| Convolution 3x3 | 1x1 stride, valid padding, outputs 13x13x64 |
| RELU            |					                             |
| Max pooling     | 2x2 stride, outputs 7x7x64                  |
| Dropout         | Keep probability 0.5                        |
| Convolution 3x3 | 1x1 stride, valid padding, outputs 5x5x128  |
| RELU            |                                             |
| Max pooling     | 2x2 stride,  outputs 3x3x128                |
| Convolution 3x3 | 1x1 stride, valid padding, outputs 1x1x256  |
| RELU            |                                             |
| Max pooling     | 2x2 stride, outputs 1x1x256                 |
| Dropout         | Keep probability 0.5                        |
| Fully connected | Inputs 3392 (7x7x64 + 256), outputs 1024    |
| RELU            |                                             |
| Dropout         | Keep probability 0.5                        |
| Fully connected | Inputs 1024, outputs 512                    |
| RELU            |	                                             |
| Dropout         | Keep probability 0.5                        |
| Fully connected | Inputs 512, outputs 43                      |
| Softmax         |                                             |

Additional maxpooling, dropout, and L2-regularization techniques have been used to reduce overfitting.

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the AdamOptimizer with a learning rate of 0.001. The epochs used was 10 while the batch size was 128. Other important parameters I tuned were the probabilty of keeping neurons for dropout, and the beta parameter for L2-regularization. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I've started from a well known architecture, LeNet-5 from Yann Le Cun, because of simplicity of implementation. It was a good starting point but it suffered from under fitting. The accuracy on both sets, training and validation, was very low (around 50%). 

So I modified the network and added more convolutional and fully-connected layers, and I used dropout as well. I've also played with the layer parameters (filter size, strides, padding, max pooling strides, fully connected outputs size) to find which changes give the best results.

But after a few runs with this new architecture I noticed that the model still tended to overfit to the original training dataset. With the best run, I obtained a 5% gap in accuracy between the training and validation sets. Hence, I added dropout between the convolution layers, and I also used L2-regularization.

With these 2 techniques, the gap between training and validation accuracies has been reduced to 3.3%. Furthermore, the accuracy increased for the validation and testing datasets. The performance of the model has been increased and generalizes better to unseen data which is the ultimate goal.

Hence, my final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.966
* test set accuracy of 0.960

Find below the learning curves. Both training and validation curves are converging toward 100%, however a gap between the two is still remaining. Future improvements could include techniques such as data augmentation to reduce overfitting even more.

![Learning curves][learning_curves]

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![New traffic signs][new_traffic_signs]

The images seem relatively similar to the training dataset. Let's see however how it is performing with them.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                 | Prediction               |
|:---------------------:|:------------------------:|
| Speed limit (30km/h)  | Stop sign                |
| Bumpy road            | U-turn                   |
| Ahead only            | Yield                    |
| **No vehicles**       | **Speed limit (80km/h)** |
| Go straight or left   | Slippery Road            |
| General caution       | Slippery Road            |

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.3%. This is less than the accuracy on the test set.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 53th cell of the Ipython notebook.

For five images, the model is relatively sure that it predicts the right sign (probabilities range from 88% to 100%). However for the only one that the model makes a wrong prediction, we can notice the model is not very sure about it (45.26%). However, the right prediction is part of the top 5 softmax probalities with 2.31% probability.

![Top 5 Softmax Probabilities][top5_softmax_prob]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

