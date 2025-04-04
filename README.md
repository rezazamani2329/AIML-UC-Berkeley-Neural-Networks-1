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

## **processin third file (multiclass_Reuters.ipynb):**
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




- 
