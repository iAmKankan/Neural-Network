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
  * <a href="https://nbviewer.org/github/iAmKankan/Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-TensorFlow-All_Codes/blob/main/Chapter%2010/2)%20Sequential%20API%20for%20Regression.ipynb">Building a Regression MLP Using the Sequential API </a>  <img src="https://img.shields.io/badge/Jupyter Notebook-FFFFFF.svg?&style=for-the-badge&logo=Jupyter&logoColor=black" align='center' width=20%/>
* [Classification MLPs](https://github.com/iAmKankan/Neural-Network/blob/main/multilayer-perceptrons.md#classification-mlps)
  * <a href="https://nbviewer.org/github/iAmKankan/Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-TensorFlow-All_Codes/blob/main/Chapter%2010/1)%20Sequential%20API%20for%20Image%20Classification.ipynb">Building an Image Classifier Using the Sequential API</a> <img src="https://img.shields.io/badge/Jupyter Notebook-FFFFFF.svg?&style=for-the-badge&logo=Jupyter&logoColor=black" align='center' width=20%/>   
  * <a href="https://nbviewer.org/github/iAmKankan/Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-TensorFlow-All_Codes/blob/main/Chapter%2010/3)%20Functional%20API%20for%20Complex%20regression%20Model.ipynb">Building Complex Models Using the Functional API</a> <img src="https://img.shields.io/badge/Jupyter Notebook-FFFFFF.svg?&style=for-the-badge&logo=Jupyter&logoColor=black" align='center' width=20%/>
* [Hyperparameter selection intutive guideline]()
  * [Number of Hidden Layers]()
  * [Number of Neurons per Hidden Layer]()
  * [Learning Rate, Batch Size, and Other Hyperparameters]()
    * [Learning rate]()
    * [Optimizer](https://github.com/iAmKankan/Neural-Network/tree/main/optimizer#readme)
    * [Batch size]()
    * [Activation function](https://github.com/iAmKankan/Neural-Network/tree/main/activation_functions)
    * [Number of iterations]()
### _References_
* [Bibliography](https://github.com/iAmKankan/Neural-Network/blob/main/multilayer-perceptrons.md#bibliography)

## The Perceptron
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* The Perceptron is one of the simplest ANN architectures, invented in 1957 by Frank Rosenblatt. 
* It is based on a slightly different artificial neuron called a **Threshold Logic Unit (TLU)**, or sometimes a **Linear Threshold Unit (LTU)**: the inputs and output are now numbers (instead of binary on/off values) and each input connection is associated with a weight.
* A single TLU can be used for simple linear binary classification. 

> #### It computes a linear combination of the inputs and if the result exceeds a threshold, it outputs the positive class or else outputs the negative class (just like a Logistic Regression classifier or a linear SVM).

<p align="center">
 <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/ArtificialNeuronModel_english.png/1024px-ArtificialNeuronModel_english.png" width=50%> 
</p>

### Training Perceptron
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* The Perceptron training algorithm proposed by Frank Rosenblatt was inspired by Hebb’s rule.
* **Donald Hebb rule** suggested that when a biological neuron often triggers another neuron, the connection between these two neurons grows stronger.

$$\Large{\color{purple} w_{i,j}^{(\textrm{Next Step})} = w_{i,j} + \eta(y_j - \hat{y_j})x_i}$$

#### Where: 
$$\Large{\color{Purple}\begin{matrix*}[l]
w_{i,j} &:& \textrm{connection weight between} \ \ i^{th} \textrm{ input neuron and } j^{th} \textrm{ output neuron} \\
x_i &:& i^{th}\textrm{ input value} \\
\hat{y_j} &:& \textrm{output of } \ j^{th}\ \textrm{ output } \\
y_j &:& \textrm{target output of }\ \ j^{th} \textrm{ output neuron} \\
\eta &:& \textrm{learning rate}
\end{matrix*}}$$

#### It can also be written as for jth element of w vector 
$$\Large{\color{Purple}\begin{matrix*}[l]
w_j & = w_j + \Delta w_j\\
& \large where,\ \Delta w_j = \eta(y^{(i)} - \hat{y_j}^{(i)})x_j^{(i)}
\end{matrix*} }$$

#### Update Weights can be written as
$$\Large{\color{Purple} W= W-\eta \frac{\partial e}{\partial w}}$$

#### Where
$$\Large{\color{Purple}\begin{matrix*}[l]
 W &=& Weight \\
 \eta &=&  Learning \ rate,\\
 \partial e  &=& Change \ in\ error, \\
 \partial w  &=& Change \ in\ weight\\
 \end{matrix*}}$$

$$\large{\color{Purple}Here \hspace{10pt} (-\eta \frac{\partial e}{\partial w}) \ = \  \Delta W, \ \ \ From \ the\ above \ \ (W_j+\Delta W_j) }$$

### Neural Network error Update Weights
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
* Suppose we have a **Neural Network** with **3 input** layers and **2 hidden** layers and we are using **sigmoid** as a **activation function** inside the **hidden layer** as well as in **final weight calculation**.

<p align="center">
  <img src="https://user-images.githubusercontent.com/12748752/138762029-20fc6d46-e47c-4131-b1d3-9ce33a3595af.png" width=50%/>
</p>

#### Input buffers(that's why not having  Bias,  Input neuron  would have  Bias),  Hidden layers,  Output Neuron are like

$$\Large{\color{Purple}\begin{matrix*}[l]
 \textrm{Input Buffer}  &=& X_1, X_2, X_3 \hspace{10pt} \textrm{ (No Bias, Input Neuron would have Bias)} \\
\textrm{W} &=& W_{i j}^{(z)} \hspace{10pt} \textrm{(i= the  destination, j= the source, z = location number)}\\
\textrm{Bias} &=& b_i \\
\textrm{Weight Summation} &=& Z_i^{(z)} \hspace{10pt} \textrm{(i= Hidden or output neuron number, z = location)}\\
\widehat{Y} &=& \textrm{\ Final output}\\
 \end{matrix*}}$$
  
#### Hidden Layer weight calculation:

$$\Large{\color{Purple} \begin{matrix*}[l]
\textrm{Hidden Layer (1)} & &\\
& Z_1^{(1)} &= W_{11}^{(1)} X_1 + W_{12}^{(1)} X_2 +  W_{13}^{(1)} X_3 +  b_{1}^{(1)}\\
& a_1^{(1)} &= \sigma(Z_1^{(1)})\\
\textrm{Hidden Layer (2)} & &\\
& Z_2^{(1)} &= W_{21}^{(1)} X_1 + W_{22}^{(1)} X_2 +  W_{23}^{(1)} X_3 +  b_{2}^{(1)}\\
& a_2^{(1)} &= \sigma(Z_2^{(1)})\\
\end{matrix*}}$$

#### Final layer weight calculation
$$\Large{\color{Purple} \begin{matrix*}[l]
\textrm{Final Output Layer} & &\\
& Z_1^{(2)} &= W_{11}^{(2)} a_1^{(1)} + W_{12}^{(2)} a_2^{(1)} +  W_{13}^{(2)} X_3 +  b_{1}^{(2)}\\
& a_1^{(2)} &= \sigma(Z_1^{(2)}) \rightarrow \hat{Y}\\
\end{matrix*}}$$

#### Weight and Bias update rule-
$$\Large{\color{Purple} \textrm{Weight Update Rule (general)}}$$

$$\Large{\color{Purple} \begin{Bmatrix*}[l]
W &=& W + \Delta W &\hspace{35pt}& b &=& b + \Delta b\\ 
\Delta W &=& - \eta\dfrac{\partial e}{\partial W} &\hspace{35pt}& \Delta b &=& - \eta\dfrac{\partial e}{\partial b}\\ 
\end{Bmatrix*}
\large\begin{matrix}
\Delta W\\
\Delta b\\
\downarrow\\
\textrm{small}\\
\end{matrix}}$$

$$\Large{\color{Purple} W = W - \eta\dfrac{\partial e}{\partial W}}$$


### Matrix representation of above diagrams

* Calculation in **Hidden layers**

$$\Large{\color{Purple} \begin{bmatrix}
W_{11}&W_{12} & W_{13} \\
W_{21}& W_{22} & W_{23}
\end{bmatrix}\_{(2 \times 3)}*
\begin{bmatrix}
X_{1} \\
X_{2} \\
X_{3} \\
\end{bmatrix}\_{(3 \times 1)} + 
\begin{bmatrix}
b_{1} \\
b_{2} \\
\end{bmatrix} = 
\begin{bmatrix} 
Z_{1} \\
Z_{2} \\
\end{bmatrix} \to 
\begin{bmatrix} 
activation(Z_{1}) \\ 
activation(Z_{2}) \\
\end{bmatrix}\to 
\begin{bmatrix} 
a_{1} \\ 
a_{2} \\
\end{bmatrix} \hspace{10pt} or \hspace{10pt} \hat{Y} }$$


* Calculation in **Output layer**

$$\Large{\color{Purple} \begin{bmatrix}
W_{11}& W_{12} \\
\end{bmatrix}\_{(1 \times 2)}*
\begin{bmatrix}
a_{1} \\
a_{2} \end{bmatrix}\_{(2 \times 1)} +
\begin{bmatrix}
b_{1} \\
\end{bmatrix} =
 \begin{bmatrix}
Z_{1} \\
\end{bmatrix}\to
\begin{bmatrix}
activation(Z_{1}) \\
\end{bmatrix} \to
\begin{bmatrix}
a_{1} \\
\end{bmatrix} \hspace{10pt} or \hspace{10pt} \hat{Y}}$$

### Practical
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)

* **Scikit-Learn** provides a **Perceptron** class that implements a single **TLU** network. 
* It can be used pretty much as you would expect—for example, on the **iris** dataset 
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

#### The signal flows only in one direction (from the inputs to the outputs), so this architecture is an example of a **Feedforward Neural Network (FNN)**.

<p align="center">
 <img src="https://user-images.githubusercontent.com/12748752/143045465-2fe26cb7-48ea-4590-b381-24215f014004.png" width=30% />
</p>

> #### When an ANN contains a deep stack of hidden layers, it is called a Deep Neural Network (DNN).

## Backpropagation - [_General Backpropagation_](https://github.com/iAmKankan/Neural-Network/blob/main/backpropagation/README.md)
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
*  **Backpropagation is simply Gradient Descent using an efficient technique for computing the gradients automatically**: 

* In just two passes through the network (one **forward**, one **backward**), the backpropagation algorithm is able to compute **the gradient of the network’s error** with regards to every single model parameter. 

> ### In other words, it can find out how each connection weight and each bias term should be tweaked in order to reduce the error. 
* Once it has these gradients, it just performs a regular Gradient Descent step, and the whole process is repeated until the network converges to the solution.

> #### Automatically computing gradients is called *automatic differentiation*, or *autodiff*. The autodiff technique used by backpropagation is called *reverse-mode autodiff*. It is fast and precise, and is well suited when the function to differentiate has many variables (e.g., connection weights) and few outputs (e.g., one loss). 

## How Backpropagation works
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
### Epochs
Backpropagation handles **_one mini-batch at a time_** (for example containing _32 instances_ each), and it goes through the full training set multiple times. Each pass is called an **epoch**. 

### Forward Pass:
**Step #1:**  Each _mini-batch_ is passed to the network’s **input layer**, which just sends it to the **first hidden layer**. 

**Step #2:** The algorithm then computes the _output of all the neurons in this layer_ (for every instance in the mini-batch). The result is passed on to the **next layer**.

**Step #3:** Again its output is computed and passed to the **next layer** and so on until we get the **_output of the last layer_** the **output layer**. 

This is the **forward pass**: it is exactly like making _predictions_, except **_all intermediate results are preserved_** since they are needed for the **backward pass**. 

### Backward Pass:
**Step #1 Loss Function:** Next, the algorithm measures the **network’s output error** (i.e., it uses a loss function that compares the desired output and the actual output of the network, and returns some measure of the error). 

**Step #2 Chain Rule:** Then it computes how much **each output connection contributed to the error**. 
* This is done analytically by simply applying the [**chain rule**](https://github.com/iAmKankan/Mathematics/blob/main/calculus/D_calculus.md#chain-rule) (perhaps the most fundamental rule in calculus), which makes this step _fast and precise_. 

**Step #3:** The algorithm then measures how much of _these error contributions_ came from each connection in the **layer below**, again using the [**chain rule**](https://github.com/iAmKankan/Mathematics/blob/main/calculus/D_calculus.md#chain-rule) — and so on until the algorithm reaches the **input layer**.
* As we explained earlier, this reverse pass efficiently measures the **error gradient** across all the connection weights in the network by propagating the error gradient backward through the network (hence the name of the algorithm).

### Finally Optimization
**Finally**, the algorithm performs a **Gradient Descent** step to tweak all the connection weights in the network, using the error gradients it just computed.

---

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


### Regression MLPs
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

* First, MLPs can be used for regression tasks. 
  * If you want to predict a single value (e.g., the price of a house given many of its features), then you just need a single output neuron: its output is the predicted value.
  * For multivariate regression (i.e., to predict multiple values at once), you need one output neuron per output dimension. 
    * For example, to locate the center of an object on an image, you need to predict 2D coordinates, so you need two output neurons. 
    * If you also want to place a bounding box around the object, then you need two more numbers: the width and the height of the object.
    *  So you end up with 4 output neurons.

> #### In general, when building an MLP for regression, you do not want to use any activation function for the output neurons, so they are free to output any range of values.


### Regression and Classification models
<p align="center">
 <img src="https://user-images.githubusercontent.com/12748752/143228831-e4318e6f-25b0-43b9-950d-a865c4df7d1c.png" width=60% height=30% />
</p>

### Typical Regression MLP Architecture

<div align="center">
 
|<ins>Hyperparameter</ins>|<ins>Typical Value</ins>|
|:-----:|:-----:|
| **Input neurons**   |    One per input feature (e.g., 28 x 28 = 784 for **MNIST**)|
|**Hidden layers** | Depends on the problem. Typically 1 to 5.|
|**Neurons  per hidden layer**| Depends on the problem. Typically 10 to 100|
|**Output neurons**| 1 per prediction dimension|
|**Hidden activation**|**ReLU** (or **SELU**, see Chapter 11)|
|**Output activation**|**None** or **ReLU**/**Softplus** (if positive outputs or **Logistic**/**Tanh** (if bounded outputs)|
|**Loss function**|**MSE** or **MAE/Huber** (if **outliers**)|

</div>

### Typical Classification MLP Architecture

<div align="center">
 
|Hyperparameter|Binary classification| Multilabel binary classification| Multiclass classification}|
|:---:|:---:|:---:|:---:|
|Input and hidden layers|Same as regression|Same as regression|Same as regression|
|Output neurons| 1 | 1 per label|1 per class|
|Output layer activation|Logistic|Logistic|Softmax|
|Loss function|Cross-Entropy|Cross-Entropy|Cross-Entropy|

</div>


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
