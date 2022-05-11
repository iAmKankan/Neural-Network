## Index

![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

## Loss Functions 
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

## Loss-Cost/Error:
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
When performing **supervised training**, a neural network’s _actual output_ must be compared against the _ideal output_ specified in the training data.<ins> _The difference between actual and ideal output is the error of the neural network_</ins>. Error calculation occurs at two levels. 
#### Local Error
First , there is the **_local error_**. This is the difference between the _**actual output**_ of _one individual neuron_ and the **_ideal output_** that was expected. The local error is calculated using an **error function**.
#### Global Error
The local errors are aggregated together to form a **_global error_**. The global error is the measurement of how well a neural network performs to the entire training set. There are several different means by which a global error can be calculated. The global error calculation methods discussed in this chapter are listed below.

###  _Generalization Error_ (or _out-of-sample error_) 
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
The only way to know how well a model will generalize to new cases is to actually try it out on new cases by split your data into two sets: **_1) the training set_** and **_2) the test set_**. 

As these names imply, you train your model using the _training set_, and you test it using the _test set_. The error rate on new cases is called the **_generalization error_** (or **out-of-sample error**), and by evaluating your model on the test set, you get an estimate of this error. _This value tells you how well your model will perform on instances it has never seen before_. 

> #### If the training error is low (i.e., your model makes few mistakes on the training set) but the generalization error is high, it means that your model is _overfitting_ the training data.

|                     **Classification**                    |                                          **Regression**                                         |
|:-----------------------------------------------------:|:-------------------------------------------------------------------------------------------:|
| **_Multiclass Classification:_**<br>Catagorical Crossentrophy            | **_Continuous Values:_**<br>Mean Squired Error<br>Mean absolute Error<br>Mean Squred Logggd Error |
| **_Binary Classification:_**<br>Binary Cross Entrophy<br>Squred Hinge | **_Discreate Values:_**<br>Poisson                                                 |


## L1 and L2 loss or Mean Absolute Error and Mean Square Error
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
*L1* and *L2* are two common loss functions in machine learning which are mainly used to minimize the error.
   1) **L1 loss function or Mean Absolute Error (MAE)**.
   2) **L2 loss function or Mean Square Error (MSE)**.

### 1) L1- _Loss function_ or _Least Absolute Deviations_(LAD) or Mean Absolute Error (MAE)
It is used to minimize the error which is the sum of all the absolute differences in between the true value and the predicted value.

<img src="https://latex.codecogs.com/svg.image?\large&space;'L1-loss'\mathrm{\&space;or\&space;MAE}\mathbf{\&space;={\color{Purple}&space;&space;\sum_{i=1}^{n}|y_{true}-y_{pridicted}|}" title="https://latex.codecogs.com/svg.image?\large 'L1-loss'\mathrm{\ or\ MAE}\mathbf{\ ={\color{Purple} \sum_{i=1}^{n}|y_{true}-y_{pridicted}|}" />

### 2) L2- _Loss Function_ or _Least square errors_(LS) or Mean Square Error (MSE)
It is also used to minimize the error which is the sum of all the squared differences in between the true value and the pedicted value.

<img src="https://latex.codecogs.com/svg.image?\large&space;'L2-loss'&space;\mathrm{\&space;or\&space;MSE}\mathbf{\&space;={\color{Purple}&space;&space;\sum_{i=1}^{n}(y_{true}-y_{pridicted})^2}" title="https://latex.codecogs.com/svg.image?\large 'L2-loss' \mathrm{\ or\ MSE}\mathbf{\ ={\color{Purple} \sum_{i=1}^{n}(y_{true}-y_{pridicted})^2}" />

**The disadvantage** of the **L2 norm** is that when there are outliers, these points will account for the main component of the loss. For example, the true value is 1, the prediction is 10 times, the prediction value is 1000 once, and the prediction value of the other times is about 1, obviously the loss value is mainly dominated by 1000.

## Huber loss
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
**Huber Loss** is often used in _regression_ problems. Compared with _L2 loss_, Huber Loss is _**less sensitive to outliers**_(because if the residual is too large, it is a piecewise function, loss is a linear function of the residual).

<img src="https://latex.codecogs.com/svg.image?\large&space;\mathbf{L\delta&space;(y,f(x))=}&space;\begin{cases}\mathbf{{\color{Purple}\frac{1}{2}(y-f(x))^2}}&&space;\mathbf{for&space;|y-f(x)|\leq&space;\delta&space;,&space;}\\\mathbf{{\color{Purple}\delta|y-f(x)|-\frac{1}{2}\delta&space;^2}}&&space;\mathbf{otherwise.}\end{cases}" title="https://latex.codecogs.com/svg.image?\large \mathbf{L\delta (y,f(x))=} \begin{cases}\mathbf{{\color{Purple}\frac{1}{2}(y-f(x))^2}}& \mathbf{for |y-f(x)|\leq \delta , }\\\mathbf{{\color{Purple}\delta|y-f(x)|-\frac{1}{2}\delta ^2}}& \mathbf{otherwise.}\end{cases}" />



## References:
* [Peltarion.com](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions)
