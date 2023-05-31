# DevnagariProject
Accessed Devanagiri handwritten characters dataset and implemented deep learning using tensorflow
1) Explored the chosen dataset and loaded them
2) Pre-processing the data
3) Built and trained Machine Learning models(VGG3, LeNet, VGG13, VGG16) which gives 99% accuracy
4) Evaluation of the model through visualization
5) Evaluation of model's performance on test data
6) Generation of classification report
7) Generation of confusion matrix

# About the dataset
* Devanagiri handwritten characters dataset consists of alphabets used in Devanagiri and I've chosen 11 characters to train my model for the project
* Total number of samples - 18,700 training samples and 3320 test samples
* Total number of measurements - Each measurement is a 32*32 grayscale image, associated with labels from 11 classes
* There are two features - Label & Image(given using pixels in .png format)
* Label - labeling images with characters
  * Label 0 - pa
  * Label 1 - na
  * Label 2 - ga
  * Label 3 - dhaa
  * Label 4 - ka
  * Label 5 - ra
  * Label 6 - taa
  * Label 7 - ja
  * Label 8 - daa
  * Label 9 - thaa
  * Label 10 - cha

# Detailed description of what I have done for this project
***1) Explored the chosen dataset and loaded them***
  * The necessary libraries were imported
  * Dataset was loaded

***2) Pre-processing the data***
  * Took empty lists for X train, y train & X test, y test as train_img, train_labels and test_img, test_labels respectively
  * Created a loop to iteratively read and store images and the corresponding character names on the above created variables
  * NP array was created for the newly formed variables
  * Exploring labels, mapping them against each character and storing them as lists
  * Convert lists into numpy arrays and check on data distribution - classes are equally distributed avoiding oversampling and undersampling 

***3) Built and trained Machine Learning models(VGG3, LeNet, VGG13, VGG16) which gives 99% accuracy***
#### Referred from https://towardsdatascience.com/simple-introduction-to-convolutional-neural-networks-cdf8d3077bac#:~:text=There%20are%20three%20types%20of,task%20on%20the%20input%20data
  * Layers used to build ML models(VGG3, LeNet, VGG13, VGG16):
    1) To begin with, I've used sequential model which is from keras and has layers stacked linearly, starting from the input, one layer is added at a time
    2) Building blocks of ConvNets, that is, convolutional layers are being inserted which possess set of independent filters and it's depth will be equal to input and other parameters are manually set. Feature maps are produced when these filters gets convolved on top of the input image
    3) Then, by adding pooling layers, dimensions were reduced transforming from higher to lower so that better performance can be achieved, thus overfitting is reduced. Acting alongside convolutional layers, complex features of an image can be learnt by them
    4) Batch normalization was added to scale down outliers so that the network learn features in a distributed method, thus not depending much on certain weights and make the model better understand images
    5) By adding dropout, some percent of neurons is dropped randomly, re-aligning weights so that overfitting can be avoided.It is a technique to regularize, penalizing parameters and is set in the range between 0.2 to 0.5
    6) To map the input to a 1D vector, flatten layer is added
    7) Dense layers are added to alter dimensions of the vector
    8) Finally, added the output layer having units equalling to number of classes used. Since I'm having multi-class classification in place, have used softmax activation function
  * Models are then compiled using sparse_categorical_crossentropy as loss function, adam as optimizer and accuracy as metrics
  * Summary of models are checked to review steps used and saved as a .npy file
  * Then models are fitted with 32 as batch_size, 6 epochs and training is started.

***4) Evaluation of the model through visualization***
  * Loss and accuracy of models were visualized in graphs by plotting training curve vs validation curve and saved as .png files. For most epochs, curves stayed close to each other and that showed they were not overfitted.   

***5) Evaluation of model's performance on test data***
  * Calculated score of models using testdata and the results were stored as .npy files, then printed them as outputs along with the error rate.

***6) Generation of classification report***
  * The precision, recall, F1 and support scores for the models are displayed in the classification report which is been saved as .npy file.

***7) Generation of confusion matrix***
#### Referred at https://python-course.eu/machine-learning/confusion-matrix-in-machine-learning.php
  * Performance of a machine learning algorithm can be measured using confusion matrix.
  * Rows represent the actual class and columns represents the predicted class, thus the intersection of both these which are away from the diagonal area represents the wrongly predicted count

# Result
All four models(VGG3, LeNet, VGG13, VGG16) almost consistently gave an accuracy of 99% with low loss and low error rates. I feel that incase if we are handling huge & complicated datasets, we can go with bigger models(VGG16, VGG13) since it has more neurons, more neural networks can be established between neurons such that the machine learns more and results would be better. For smaller datasets, we can stick to the smaller models(VGG3, LeNet).
