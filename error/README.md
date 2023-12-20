## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* [Loss Functions](#loss-functions)
* [Loss-Cost/Error](#loss-costerror)
* [Generalization Error (or out-of-sample error)](#generalization-error-or-out-of-sample-error)
#### ◼️ Regression Loss
* [L1- Loss function or Least Absolute Deviations(LAD) and Mean Absolute Error (MAE)](#l1--loss-function-or-least-absolute-deviationslad-and-mean-absolute-error-mae)
* [L2- Loss Function or Least square errors(LS) and Mean Square Error (MSE)](#l2--loss-function-or-least-square-errorsls-and-mean-square-error-mse)
* [Huber loss](#huber-loss)
* [Hinge Loss](#hinge-loss)
#### ◼️ Classification Loss
* [Categorical Crossentropy](#categorical-crossentropy)
* [Binary crossentropy](#binary-crossentropy)

#### ◼️ [References](#references)

### ⬛ Loss Functions 
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
**Loss function** is a method of evaluating **_how well your algorithm models your data set_**. Better the prediction is lesser the loss you will get.

The loss error is computed for a single training example. If we have ‘**m**’ number of examples then the average of the loss function of the entire training set is called **Cost function**.
#### <ins>_Cost function (J) = (Sum of Loss error for ‘m’ examples) / m_</ins>

Depending on the problem Cost Function can be formed in many different ways.

The purpose of Cost Function is to be either:
* **Minimized** — then returned value is usually called cost, loss or error. The goal is to find the values of model parameters for which Cost Function return as small number as possible.
* **Maximized** — then the value it yields is named a reward. The goal is to find values of model parameters for which returned number is as large as possible.

In other words, the terms cost and loss functions almost refer to the same meaning. But, the loss function _mainly applies for a single training set as compared to the cost function which deals with a penalty for a number of training sets or the complete batch_. It is also sometimes called an error function. In short, we can say that the loss function is a part of the cost function. 

The cost function is calculated as an average of loss functions. The loss function is a value that is calculated at every instance. So, for a single training cycle loss is calculated numerous times, but the cost function is only calculated once.

#### Example 1:
* One of the loss function used in Linear Regression, the **square loss**
* One of the cost function used in Linear Regression, the **Mean Squared Error**

#### Example 2:
* One of the loss function used in SVM, the **hinge loss**
* **SVM cost function**

### ⬛ Loss-Cost/Error:
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
When performing **supervised training**, a neural network’s _actual output_ must be compared against the _ideal output_ specified in the training data.<ins> _The difference between actual and ideal output is the error of the neural network_</ins>. Error calculation occurs at two levels. 
### 🔲 Local Error
First , there is the **_local error_**. This is the difference between the _**actual output**_ of _one individual neuron_ and the **_ideal output_** that was expected. The local error is calculated using an **error function**.
### 🔲 Global Error
The local errors are aggregated together to form a **_global error_**. The global error is the measurement of how well a neural network performs to the entire training set. There are several different means by which a global error can be calculated. The global error calculation methods discussed in this chapter are listed below.

### 🔲 _Generalization Error_ (or _out-of-sample error_) 
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
The only way to know how well a model will generalize to new cases is to actually try it out on new cases by split your data into two sets: **_1) the training set_** and **_2) the test set_**. 

As these names imply, you train your model using the _training set_, and you test it using the _test set_. The error rate on new cases is called the **_generalization error_** (or **out-of-sample error**), and by evaluating your model on the test set, you get an estimate of this error. _This value tells you how well your model will perform on instances it has never seen before_. 

> #### If the training error is low (i.e., your model makes few mistakes on the training set) but the generalization error is high, it means that your model is _overfitting_ the training data.

   
<p align="center">
 <img src="https://github.com/iAmKankan/Neural-Network/assets/12748752/ccb39acf-ed8d-4523-9075-488bd163fc61" width=60%/>
   <br>
<ins><b> Different loss</b></ins>
</p>


### ⬛ Regression Losses:
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)

### 🔲 1. <ins> L2-Loss</ins> or <ins>Least Square Errors(LS)</ins> and <ins>Mean Square Error(MSE) </ins>
It is also used to minimize the error which is the sum of all the squared differences in between the true value and the pedicted value.
### ♠️ Math equation for _L2 loss_:
<img src="https://latex.codecogs.com/svg.image?\large&space;&space;\mathrm{L2-loss}\mathbf{\&space;={\color{Purple}&space;\sum_{i=1}^{n}(y_{true}-y_{pridicted})^2}" title="https://latex.codecogs.com/svg.image?\large \mathrm{L2-loss}\mathbf{\ ={\color{Purple} \sum_{i=1}^{n}(y_{true}-y_{pridicted})^2}" />

### ♠️ Math equation for  _Mean Square Error (MSE)_:
<img src="https://latex.codecogs.com/svg.image?\large&space;&space;\mathrm{MSE}\mathbf{\&space;={\color{Purple}\frac{1}{n}&space;\sum_{i=1}^{n}(y_{i}-\hat{y_i})^2}" title="https://latex.codecogs.com/svg.image?\large \mathrm{MSE}\mathbf{\ ={\color{Purple}\frac{1}{n} \sum_{i=1}^{n}(y_{i}-\hat{y_i})^2}" />

### ♠️ <ins>Disadvantages:</ins>
**The disadvantage** of the **L2 norm** is that when there are outliers, these points will account for the main component of the loss. For example, the true value is 1, the prediction is 10 times, the prediction value is 1000 once, and the prediction value of the other times is about 1, obviously the loss value is mainly dominated by 1000.


### L1 and L2 loss or Mean Absolute Error and Mean Square Error
*L1* and *L2* are two common loss functions in machine learning which are mainly used to minimize the error.
   1) **L1 loss function and Mean Absolute Error (MAE)**.
   2) **L2 loss function or Mean Square Error (MSE)**.

## L1- _Loss function_ or _Least Absolute Deviations_(LAD) and _Mean Absolute Error (MAE)_
It is used to minimize the error which is the sum of all the absolute differences in between the true value and the predicted value.
### Math equation for _L1 loss_:

$$\Large \mathrm{\ L1-loss}\mathbf{\ ={\color{Purple} \sum_{i=1}^{n}|y_{true}-y_{pridicted}|} }$$

### Math equation for  _Mean Absolute Error (MAE)_:

$$\Large \mathrm{\ MAE}\mathbf{\ ={\color{Purple}\frac{1}{n} \sum_{i=1}^{n}|y_{i}-\hat{y_{i}}|}}$$


## Huber loss
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
**Huber Loss** is often used in _regression_ problems. Compared with _L2 loss_, Huber Loss is _**less sensitive to outliers**_(because if the residual is too large, it is a piecewise function, loss is a linear function of the residual).

<img src="https://latex.codecogs.com/svg.image?\large&space;\mathbf{L\delta&space;(y,f(x))=}&space;\begin{cases}\mathbf{{\color{Purple}\frac{1}{2}(y-f(x))^2}}&&space;\mathbf{for&space;|y-f(x)|\leq&space;\delta&space;,&space;}\\\mathbf{{\color{Purple}\delta|y-f(x)|-\frac{1}{2}\delta&space;^2}}&&space;\mathbf{otherwise.}\end{cases}" title="https://latex.codecogs.com/svg.image?\large \mathbf{L\delta (y,f(x))=} \begin{cases}\mathbf{{\color{Purple}\frac{1}{2}(y-f(x))^2}}& \mathbf{for |y-f(x)|\leq \delta , }\\\mathbf{{\color{Purple}\delta|y-f(x)|-\frac{1}{2}\delta ^2}}& \mathbf{otherwise.}\end{cases}" />
Among them, **&delta;** is a set parameter, **y** represents the real value, and _**f(x)**_ represents the predicted value.

The advantage of this is that when the residual is small, the loss function is L2 norm, and when the residual is large, it is a linear function of L1 norm

## Hinge Loss

Hinge loss is often used for binary classification problems, such as ground true: t = 1 or -1, predicted value y = wx + b

In the svm classifier, the definition of hinge loss is

In other words, the closer the y is to t, the smaller the loss will be.

## Cross-entropy loss
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)

## Categorical Crossentropy
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
**Categorical crossentropy** is a loss function that is used in **_multi-class classification_** tasks. **_These are tasks where an example can only belong to one out of many possible categories, and the model must decide which one._**

Formally, it is designed to quantify the difference between two probability distributions.

### Math equation:
<img src="https://latex.codecogs.com/svg.image?\large&space;Loss\&space;\mathbf{={\color{Purple}&space;-&space;\sum_{i=1}^{Output\&space;size}y_i.&space;\log&space;\hat{y_i}}}" title="https://latex.codecogs.com/svg.image?\large Loss\ \mathbf{={\color{Purple} - \sum_{i=1}^{Output\ size}y_i. \log \hat{y_i}}}" />

### How to use Categorical Crossentropy
The categorical crossentropy is well suited to **classification tasks**, since one example can be considered to belong to a specific category with probability **1**, and to other categories with probability **0**.

**Example:** The MNIST number recognition tutorial, where you have images of the digits 0, 1, 2, 3, 4, 5, 6, 7, 8, and 9.
The model uses the categorical crossentropy to learn to give a high probability to the correct digit and a low probability to the other digits.

### Activation functions
#### **_Softmax_** is the only activation function recommended to use with the _categorical crossentropy loss function_.

Strictly speaking, the output of the model only needs to be positive so that the logarithm of every output value <img src="https://latex.codecogs.com/svg.image?\mathrm{{\color{Purple}\hat{y_i}}}" title="https://latex.codecogs.com/svg.image?\mathrm{{\color{Purple}\hat{y_i}}}" align="center" /> exists. However, the main appeal of this loss function is for comparing two probability distributions. The **softmax** activation rescales the model output so that it has the right properties.

## Binary crossentropy
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
**Binary crossentropy** is a loss function that is used in **_binary classification_** tasks. These are tasks that answer a question with only two choices (**yes** or **no**, **A** or **B**, **0** or **1**, **left** or **right**). Several independent such questions can be answered at the same time, as in **_multi-label classification_** or in **_binary image segmentation_**.

Formally, this loss is equal to the average of the categorical crossentropy loss on many two-category tasks.

### Math equation:
<img src="https://latex.codecogs.com/svg.image?\large&space;Loss\&space;\mathrm{={\color{Purple}&space;-&space;\frac{1}{output\&space;size}}}\mathbf{{\color{Purple}&space;\sum_{i=1}^{output\&space;size}y_i.&space;\log&space;\hat{y_i}&space;&plus;&space;(1-y_i).\log(1-\hat{y_i})}}" title="https://latex.codecogs.com/svg.image?\large Loss\ \mathrm{={\color{Purple} - \frac{1}{output\ size}}}\mathbf{{\color{Purple} \sum_{i=1}^{output\ size}y_i. \log \hat{y_i} + (1-y_i).\log(1-\hat{y_i})}}" />

### How to use binary crossentropy
The binary crossentropy is very convenient to train a model to solve many _classification_ problems at the same time, if each classification can be reduced to a binary choice (i.e. **yes or no**, **A or B**, **0 or 1**).

**Example**: The build your own music critic tutorial contains music data and 46 labels like Happy, Hopeful, Laid back, Relaxing etc.
The model uses the binary crossentropy to learn to tag songs with every applicable label.

### Activation functions
#### **_Sigmoid_** _is the only activation function compatible with the_ **_binary crossentropy loss function_**. You must use it on the **last block** before the target block.

The binary crossentropy needs to compute the logarithms of <img src="https://latex.codecogs.com/svg.image?\mathrm{{\color{Purple}\hat{y_i}}}" title="https://latex.codecogs.com/svg.image?\mathrm{{\color{Purple}\hat{y_i}}}" align="center" /> and <img src="https://latex.codecogs.com/svg.image?\mathrm{{\color{Purple}(1-\hat{y_i})}}" title="https://latex.codecogs.com/svg.image?\mathrm{{\color{Purple}(1-\hat{y_i})}}" align="center"/>, which only exist if <img src="https://latex.codecogs.com/svg.image?\mathrm{{\color{Purple}\hat{y_i}}}" title="https://latex.codecogs.com/svg.image?\mathrm{{\color{Purple}\hat{y_i}}}" align="center" /> is between **0** and **1**. 

The **sigmoid activation function** is the only one to **guarantee that independent outputs lie within this range**.

## References:
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* [Peltarion.com](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions)
