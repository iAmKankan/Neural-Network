## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
### _Perceptron_
* [The Perceptron](#the-perceptron)
* [Training Perceptron](#training-perceptron)
### _Multi-Layer Perceptron_
* [Multi-Layer Perceptron](#multi-layer-perceptron)
* [Backpropagation](#backpropagation)
  * [How Backpropagation works](#how-backpropagation-works)
* [Regression MLPs](https://github.com/iAmKankan/Neural-Network/blob/main/multilayer-perceptrons.md#regression-mlps)
  * <a href="https://nbviewer.org/github/iAmKankan/Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-TensorFlow-All_Codes/blob/main/Chapter%2010/2)%20Sequential%20API%20for%20Regression.ipynb">Building a Regression MLP Using the Sequential API</a>  
* [Classification MLPs](https://github.com/iAmKankan/Neural-Network/blob/main/multilayer-perceptrons.md#classification-mlps)
  * <a href="https://nbviewer.org/github/iAmKankan/Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-TensorFlow-All_Codes/blob/main/Chapter%2010/1)%20Sequential%20API%20for%20Image%20Classification.ipynb">Building an Image Classifier Using the Sequential API</a>    
  * <a href="https://nbviewer.org/github/iAmKankan/Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-TensorFlow-All_Codes/blob/main/Chapter%2010/3)%20Functional%20API%20for%20Complex%20regression%20Model.ipynb">Building Complex Models Using the Functional API</a>
* [Hyperparameter selection intutive guideline]()
  * [Number of Hidden Layers]()
  * [Number of Neurons per Hidden Layer]()
  * [Learning Rate, Batch Size, and Other Hyperparameters]()
    * [Learning rate]()
    * [Optimizer]()
    * [Batch size]()
    * [Activation function]()
    * [Number of iterations]()
### _Code_


### _References_
* [Bibliography](https://github.com/iAmKankan/Neural-Network/blob/main/multilayer-perceptrons.md#bibliography)

## The Perceptron
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* The Perceptron is one of the simplest ANN architectures, invented in 1957 by Frank Rosenblatt. 
* It is based on a slightly different artificial neuron called a **Threshold Logic Unit (TLU)**, or sometimes a **Linear Threshold Unit (LTU)**: the inputs and output are now numbers (instead of binary on/off values) and each input connection is associated with a weight.
* A single TLU can be used for simple linear binary classification. 
> * It computes a linear combination of the inputs and if the result exceeds a threshold, it outputs the positive class or else outputs the negative class (just like a Logistic Regression classifier or a linear SVM).

### Training Perceptron
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* The Perceptron training algorithm proposed by Frank Rosenblatt was inspired by Hebb’s rule.
* **Donald Hebb rule** suggested that when a biological neuron often triggers another neuron, the connection between these two neurons grows stronger.

> #### <img src="http://latex.codecogs.com/svg.image?w_{i,j}^{(Next\&space;Step)}&space;=&space;w_{i,j}&space;&plus;&space;\eta(y_j&space;-&space;\hat{y_j})x_i" title="w_{i,j}^{(Next\ Step)} = w_{i,j} + \eta(y_j - \hat{y_j})x_i" width=45% />

> #### Where
> 
>> <img src="https://latex.codecogs.com/svg.image?&space;w_{i,j}&space;\textrm{&space;:&space;connection&space;weight&space;between}&space;\&space;\&space;i^{th}&space;&space;\textrm{input&space;neuron&space;and&space;}&space;j^{th}&space;&space;\textrm{&space;output&space;neuron}" title=" w_{i,j} \textrm{ : connection weight between} \ \ i^{th} \textrm{input neuron and } j^{th} \textrm{ output neuron}" />.  
>>
>> <img src="https://latex.codecogs.com/svg.image?x_i&space;:&space;i^{th}\textrm{&space;input&space;value}" title="x_i : i^{th}\textrm{ input value}" />.
>>
>> <img src="https://latex.codecogs.com/svg.image?\hat{y_j}&space;:&space;\textrm{output&space;of}&space;\&space;j^{th}\&space;\textrm{&space;output&space;}" title="\hat{y_j} : \textrm{output of} \ j^{th}\ \textrm{ output }" />.
>>
>> <img src="https://latex.codecogs.com/svg.image?y_j&space;:&space;\textrm{target&space;output&space;of}\&space;\&space;j^{th}&space;\textrm{&space;output&space;neuron}" title="y_j : \textrm{target output of}\ \ j^{th} \textrm{ output neuron}" />.
>>
>> <img src="https://latex.codecogs.com/svg.image?\eta&space;:&space;\textrm{learning&space;rate}" title="\eta : \textrm{learning rate}" />.  

> #### It can also be written as for jth element of w vector 
> <img src="https://latex.codecogs.com/svg.image?w_j&space;=&space;w_j&space;&plus;&space;\triangle&space;w_j" title="w_j = w_j + \triangle w_j" />.
>
> <img src="https://latex.codecogs.com/svg.image?where,\&space;\triangle&space;w_j&space;=&space;&space;\eta(y^{(i)}&space;-&space;\hat{y_j}^{(i)})x_j^{(i)}" title="where,\ \triangle w_j = \eta(y^{(i)} - \hat{y_j}^{(i)})x_j^{(i)}" />.




* Scikit-Learn provides a Perceptron class that implements a single TLU network. 
* It can be used pretty much as you would expect—for example, on the iris dataset 
```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
iris = load_iris()
X = iris.data[:, (2, 3)] # petal length, petal width
y = (iris.target == 0).astype(np.int) # Iris Setosa?
per_clf = Perceptron()
per_clf.fit(X, y)
y_pred = per_clf.predict([[2, 0.5]])
```
## Multi-Layer Perceptron
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* An MLP is composed of -
   * **One (passthrough) input layer,**
   * **One or more layers of TLUs (Threshold Logic Unit)- called hidden layers,**
   * **One final layer of TLU (Threshold Logic Unit) called the output layer.**

* The layers close to the input layer are usually called the **lower layers**, and the ones close to the outputs are usually called the **upper layers**. 
* Every layer except the output layer includes **a bias neuron** and is fully connected to the next layer.

> #### The signal flows only in one direction (from the inputs to the outputs), so this architecture is an example of a **Feedforward Neural Network (FNN)**.

<img src="https://user-images.githubusercontent.com/12748752/143045465-2fe26cb7-48ea-4590-b381-24215f014004.png" width=30% />


> #### When an ANN contains a deep stack of hidden layers, it is called a Deep Neural Network (DNN).

## Backpropagation 
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
*  **Backpropagation is simply Gradient Descent using an efficient technique for computing the gradients automatically**: 

* In just two passes through the network (one **forward**, one **backward**), the backpropagation algorithm is able to compute **the gradient of the network’s error** with regards to every single model parameter. 

> ### In other words, it can find out how each connection weight and each bias term should be tweaked in order to reduce the error. 
* Once it has these gradients, it just performs a regular Gradient Descent step, and the whole process is repeated until the network converges to the solution.

> #### Automatically computing gradients is called *automatic differentiation*, or *autodiff*. The autodiff technique used by backpropagation is called *reverse-mode autodiff*. It is fast and precise, and is well suited when the function to differentiate has many variables (e.g., connection weights) and few outputs (e.g., one loss). 

### How Backpropagation works
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

* It handles one mini-batch at a time (for example containing 32 instances each), and it goes through the full training set multiple times. Each pass is called an **epoch**. 
* Each mini-batch is passed to the network’s input layer, which just sends it to the first hidden layer. The algorithm then computes the output of all the neurons in this layer (for every instance in the mini-batch). The result is passed on to the next layer, its output is computed and passed to the next layer, and so on until we get the output of the last layer, the output layer. This is the forward pass: it is exactly like making predictions, except all intermediate results are preserved since they are needed for the backward pass. 
* Next, the algorithm measures the network’s output error (i.e., it uses a loss function that compares the desired output and the actual output of the network, and returns some measure of the error). 
* Then it computes how much each output connection contributed to the error. This is done analytically by simply applying the [chain rule](https://github.com/iAmKankan/Mathematics/blob/main/D_calculus.md#chain-rule) (perhaps the most fundamental rule in calculus), which makes this step fast and precise. 
* The algorithm then measures how much of these error contributions came from each connection in the layer below, again using the [chain rule](https://github.com/iAmKankan/Mathematics/blob/main/D_calculus.md#chain-rule) —and so on until the algorithm reaches the input layer. As we explained earlier, this reverse pass efficiently measures the error gradient across all the connection weights in the network by propagating the error gradient backward through the network (hence the name of the algorithm).
* Finally, the algorithm performs a Gradient Descent step to tweak all the connection weights in the network, using the error gradients it just computed.

> ### Summary:
> * For each training instance the backpropagation algorithm goes steps like-
>   * **First makes a prediction (forward pass),**
>   * **Measures the error,**
>   * **Then goes through each layer in reverse to measure the error contribution from each connection (reverse pass),** 
>   * **Finally slightly tweaks the connection weights to reduce the error (Gradient Descent step).**

---
> ### It is important to initialize all the hidden layers’ connection weights randomly. 
> * For example, if you initialize all weights and biases to zero, then all neurons in a given layer will be perfectly identical, and thus backpropagation will affect them in exactly the same way, so they will remain identical. 
> * In other words, despite having hundreds of neurons per layer, your model will act as if it had only one neuron per layer: it won’t be too smart. 
> * If instead you randomly initialize the weights, you break the symmetry and allow backpropagation to train a diverse team of neurons.


## Regression MLPs
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

* First, MLPs can be used for regression tasks. 
  * If you want to predict a single value (e.g., the price of a house given many of its features), then you just need a single output neuron: its output is the predicted value.
  * For multivariate regression (i.e., to predict multiple values at once), you need one output neuron per output dimension. 
    * For example, to locate the center of an object on an image, you need to predict 2D coordinates, so you need two output neurons. 
    * If you also want to place a bounding box around the object, then you need two more numbers: the width and the height of the object.
    *  So you end up with 4 output neurons.

> #### In general, when building an MLP for regression, you do not want to use any activation function for the output neurons, so they are free to output any range of values.

> ### Typical Regression MLP Architecture

<img src="https://latex.codecogs.com/svg.image?\begin{matrix}\\\textbf{Hyperparameter}&space;&&space;\textbf{Typical&space;Value}\\\mathrm{input\&space;neurons}&space;&&space;\mathrm{One\&space;per\&space;input\&space;feature\&space;(e.g.,\&space;28\&space;x\&space;28\&space;=\&space;784\&space;for\&space;MNIST)}\\\mathrm{hidden\&space;layers}&space;&&space;\mathrm{Depends\&space;on\&space;the\&space;problem.\&space;Typically\&space;1\&space;to\&space;5.}\\\mathrm{neurons\&space;per\&space;hidden\&space;layer}&space;&&space;\mathrm{Depends\&space;on\&space;the\&space;problem.\&space;Typically\&space;10\&space;to\&space;100.}\\\mathrm{output\&space;neurons}&space;&&space;\mathrm{1\&space;per\&space;prediction\&space;dimension}\\\mathrm{Hidden\&space;activation}&space;&&space;\mathrm{ReLU\&space;(or\&space;SELU,\&space;see\&space;Chapter\&space;11)}\\\mathrm{Output\&space;activation}&space;&&space;\mathrm{None\&space;or\&space;ReLU/Softplus\&space;(if\&space;positive\&space;outputs)\&space;or\&space;Logistic/Tanh\&space;(if\&space;bounded\&space;outputs)}\\\mathrm{Loss\&space;function}&space;&&space;\mathrm{MSE\&space;or\&space;MAE/Huber\&space;(if\&space;outliers)}\\\end{matrix}" title="\begin{matrix}\\\textbf{Hyperparameter} & \textbf{Typical Value}\\\mathrm{input\ neurons} & \mathrm{One\ per\ input\ feature\ (e.g.,\ 28\ x\ 28\ =\ 784\ for\ MNIST)}\\\mathrm{hidden\ layers} & \mathrm{Depends\ on\ the\ problem.\ Typically\ 1\ to\ 5.}\\\mathrm{neurons\ per\ hidden\ layer} & \mathrm{Depends\ on\ the\ problem.\ Typically\ 10\ to\ 100.}\\\mathrm{output\ neurons} & \mathrm{1\ per\ prediction\ dimension}\\\mathrm{Hidden\ activation} & \mathrm{ReLU\ (or\ SELU,\ see\ Chapter\ 11)}\\\mathrm{Output\ activation} & \mathrm{None\ or\ ReLU/Softplus\ (if\ positive\ outputs)\ or\ Logistic/Tanh\ (if\ bounded\ outputs)}\\\mathrm{Loss\ function} & \mathrm{MSE\ or\ MAE/Huber\ (if\ outliers)}\\\end{matrix}" />



## Classification MLPs
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

> ### Typical Classification MLP Architecture

<img src="https://latex.codecogs.com/svg.image?\begin{matrix}\\\textbf{Hyperparameter}&space;&&space;\textbf{Binary\&space;classification}&&space;\textbf{Multilabel\&space;binary\&space;classification}&&space;\textbf{Multiclass\&space;classification}\\\mathrm{Input\&space;and\&space;hidden\&space;layers}&space;&&space;\mathrm{Same\&space;as\&space;regression}&space;&&space;\mathrm{Same\&space;as\&space;regression}&space;&&space;\mathrm{Same\&space;as\&space;regression}\\\mathrm{output\&space;neurons}&&space;1&space;&&space;\mathrm{1\&space;per\&space;label}&space;&&space;\mathrm{1\&space;per\&space;class}\\&space;\mathrm{Output\&space;layer\&space;activation}&&space;\mathrm{Logistic}&space;&&space;\mathrm{Logistic}&space;&&space;\mathrm{Softmax}\\\mathrm{Loss\&space;function}&space;&&space;\mathrm{Cross-Entropy}&space;&&space;\mathrm{Cross-Entropy}&space;&&space;\mathrm{Cross-Entropy}\end{matrix}" title="\begin{matrix}\\\textbf{Hyperparameter} & \textbf{Binary\ classification}& \textbf{Multilabel\ binary\ classification}& \textbf{Multiclass\ classification}\\\mathrm{Input\ and\ hidden\ layers} & \mathrm{Same\ as\ regression} & \mathrm{Same\ as\ regression} & \mathrm{Same\ as\ regression}\\\mathrm{output\ neurons}& 1 & \mathrm{1\ per\ label} & \mathrm{1\ per\ class}\\ \mathrm{Output\ layer\ activation}& \mathrm{Logistic} & \mathrm{Logistic} & \mathrm{Softmax}\\\mathrm{Loss\ function} & \mathrm{Cross-Entropy} & \mathrm{Cross-Entropy} & \mathrm{Cross-Entropy}\end{matrix}" />

![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

<img src="https://user-images.githubusercontent.com/12748752/143228831-e4318e6f-25b0-43b9-950d-a865c4df7d1c.png" width=60% height=30% />


## Hyperparameter tuning for Neural Networks
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

>  #### Problem:
*  How do you know what combination of hyperparameters is the best for your task?
* In a simple MLP you can change the number of 
   * layers,
   * the number of neurons per layer,
   * the type of activation function to use in each layer, 
   * the weight initialization logic, and much more.

> #### Solution 01
* One option is to simply try many combinations of hyperparameters and see which one works best on the validation set (or use K-fold crossvalidation). 
* For example, we can use `GridSearchCV` or `RandomizedSearchCV` to explore the hyperparameter space.
* Both are achievable but very slow.
 

> #### Solution 02
* Here are some Python libraries you can use to optimize hyperparameters:
  * Hyperopt
  * Hyperas, kopt, or Talos
  * Keras Tuner
  * Scikit-Optimize (skopt)
  * Spearmint
  * Hyperband
  * Sklearn-Deap
  
### Hyperparameter selection intutive guideline
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

### Number of Hidden Layers
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

* For many problems you can start with just one or two hidden layers and the neural network will work just fine.
* For more complex problems, you can ramp up the number of hidden layers until you start overfitting the training set.
### Number of Neurons per Hidden Layer
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

* The number of neurons in the input and output layers is determined by the type of input and output your task requires.
* For example, the MNIST task requires 28 × 28 = 784 input neurons and 10 output neurons.
* As for the hidden layers, it used to be common to size them to form a pyramid, with fewer and fewer neurons at each layer

> #### In general you will get more bang for your buck by increasing the number of layers instead of the number of neurons per layer.

### Learning Rate, Batch Size, and Other Hyperparameters
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
> ### Learning rate
> ### Optimizer
> ### Batch size
* In practice, large batch sizes often lead to training instabilities, especially at the beginning of training, and the resulting model may not generalize as well as a model trained with a small batch size. 
* In April 2018, Yann LeCun even tweeted "Friends don’t let friends use mini-batches larger than 32"
> ### Activation function
* In general, the ReLU activation function will be a good default for all hidden layers. For the output layer, it really depends on your task.
> ### Number of iterations
* In most cases, the number of training iterations does not actually need to be tweaked: just use early stopping instead

> #### The optimal learning rate depends on the other hyperparameters—especially the batch size—so if you modify any hyperparameter, make sure to update the learning rate as well.

## Bibliography
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition by Aurélien Géron**
