# UC-Berkeley--Neural-Networks

## in this project we apply neural networks in differnet datasets and check their performance and prediction. 
**Moreover, we apply LSTM for time series**

## process of neural networks in **first file (AIML UC Berkeley neural_network1_UC_Berkelry_.ipynb):**

#### **part one**

- define visualization decision boundaries at first


- using tensorflow, keras and layers to create model with Sequential

- use Dense to create layers

- Apply compile for model

- Fit the model
- Visualize the model for the first time
- create a more complex neural network, we'll require more epochs to train.
- we also use 16 neuron i neach internal layer

#### **Part Two- Using a Neural Net for Non-Concentric Data**
- Check accuracy and loss
- The Network Architecture
- check the optimum epoch are needed with visualization for loss function

#### **Part Three- Full Network Example**
- Preparing the features
- The Network Architecture
- Train and Evaluate the Network

#### **Part Four- Multiclass and NN**
- we use 'softmax' as activation for output to show we are working with multiclasses
- Another example of multi class for NN with encoders
- Define The Neural Network Model
- we define 'softmax' as acivation to show are working with multiclasses, we have three layers for output and 5 layer for internal netwroks

 **Another example of multi class for NN with encoders (here is to_categorical)**
- Preparing the data
- The Network Architecture
- Compiling the model
- Fitting the model

## **process in second file (deep_neural_networks1.ipynb):**
- 2D Corner Data
- Multiclass Example
- Is the model overfit?

## **process in third file (multiclass_Reuters.ipynb):**
- classify Reuters newswires into 6 mutually exclusive topics
- instance of single-label multiclass classification. If each data point could belong to multiple categories (in this case, topics), we’d be facing a multilabel multiclass classification problem.

- Reuters dataset
- Decode it back to words
- Preparing the data
- Building your model
- Compile the model
- Validating set
- Train
- accuracy of a random classifier
- Generating predictions on new data
- A different way to handle the labels and the loss
- The importance of having sufficiently large intermediate layers
- Further experiments
-**Wrapping up**
 - Here’s what you should take away from this example:

-If you’re trying to classify data points among N classes, your model should end with a Dense layer of size N.

-In a single-label, multiclass classification problem, your model should end with a softmax activation so that it will output a probability distribution over the N output classes.

- Categorical crossentropy is almost always the loss function you should use for such problems. It minimizes the distance between the probability distributions output by the model and the true distribution of the targets.
- There are two ways to handle labels in multiclass classification:
  - Encoding the labels via categorical encoding (also known as one-hot encoding) and using categorical_crossentropy as a loss function
  - Encoding the labels as integers and using the sparse_categorical_crossentropy loss function
- If you need to classify data into a large number of categories, you should avoid creating information bottlenecks in your model due to intermediate layers that are too small.


## **process in the 4th file(multiclass_iris.ipynb)**
### Iris dataset
- extract input and output columns
- encode class values as integers
- Define The Neural Network Model
- accuract and loss

 ## **Process in the 5th fiel (multiclass_mnist.ipynb)**
 
- #### Load the dataset
**tf.keras** provides a set of convenience functions for loading well-known datasets. Each of these convenience functions does the following:

Loads both the training set and the test set.
Separates each set into features and labels.
The relevant convenience function for MNIST is called **mnist.load_data()**:
- View the dataset
- Task 1: Normalize feature values
- Define a plotting function
- Invoke the previous functions
- Task 2: Optimize the model

## **Process in the 6th file (timeseries.ipynb)**

- Other things we can do with time series
- Classification—Assign one or more categorical labels to a timeseries. For instance, given the timeseries of the activity of a visitor on a website, classify whether the visitor is a bot or a human.
- Event detection—Identify the occurrence of a specific expected event within a continuous data stream. A particularly useful application is “hotword detection,” where a model monitors an audio stream and detects utterances like “Ok Google” or “Hey Alexa.”
- Anomaly detection—Detect anything unusual happening within a continuous datastream. Unusual activity on your corporate network? Might be an attacker. Unusual readings on a manufacturing line? Time for a human to go take a look. Anomaly detection is typically done via unsupervised learning, because you often don’t know what kind of anomaly you’re looking for, so you can’t train on specific anomaly examples.
- **climate dataset**
- Preparing the data
given data covering the previous five days and sampled once per hour, can we predict the temperature in 24 hours?

the data is already numerical, so you don’t need to do any vectorization

But each timeseries in the data is on a different scale (for example, atmospheric pressure, measured in mbar, is around 1,000, while H2OC, measured in millimoles per mole, is around 3). We’ll normalize each timeseries independently so that they all take small values on a similar scale. We’re going to use the first 210,225 timesteps as training data, so we’ll compute the mean and standard deviation only on this fraction of the data.

-Sampling the data
-Predict last value
- basic machine learning model
a fully connected model that starts by flattening the data and then runs it through two Dense layers.
Note the lack of an activation function on the last Dense layer, which is typical for a regression problem.
We use mean squared error (MSE) as the loss, rather than MAE, because unlike MAE, it’s smooth around zero, which is a useful property for gradient descent. We will monitor MAE by adding it as a metric in compile().

-**try a 1D convolutional model**
 -A first recurrent baseline
- Simple LSTM examplem


