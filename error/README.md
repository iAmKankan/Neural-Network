## Index

![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
## Error:
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
When performing **supervised training**, a neural network’s _actual output_ must be compared against the _ideal output_ specified in the training data.<ins> _The difference between actual and ideal output is the error of the neural network_</ins>. Error calculation occurs at two levels. 

**First**, there is the **_local error_**. This is the difference between the _**actual output**_ of _one individual neuron_ and the **_ideal output_** that was expected. The local error is calculated using an **error function**.

The local errors are aggregated together to form a **_global error_**. The global error is the measurement of how well a neural network performs to the entire training set. There are several different means by which a global error can be calculated. The global error calculation methods discussed in this chapter are listed below.


###  _Generalization Error_ (or _out-of-sample error_) 
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
The only way to know how well a model will generalize to new cases is to actually try it out on new cases by split your data into two sets: **_1) the training set_** and **_2) the test set_**. 

As these names imply, you train your model using the _training set_, and you test it using the _test set_. The error rate on new cases is called the **_generalization error_** (or **out-of-sample error**), and by evaluating your model on the test set, you get an estimate of this error. _This value tells you how well your model will perform on instances it has never seen before_. 

> #### If the training error is low (i.e., your model makes few mistakes on the training set) but the generalization error is high, it means that your model is _overfitting_ the training data.

## L1 and L2 loss
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
*L1* and *L2* are two common loss functions in machine learning which are mainly used to minimize the error.
   1) **L1 loss function** are also known as **Least Absolute Deviations** in short **LAD**.
   2) **L2 loss function** are also known as **Least square errors** in short **LS**.

### 1) L1 Loss function
It is used to minimize the error which is the sum of all the absolute differences in between the true value and the predicted value.

<img src="https://latex.codecogs.com/svg.image?\large&space;\mathrm{L1\&space;loss}\mathbf{\&space;={\color{Purple}&space;&space;\sum_{i=1}^{n}|y_{true}-y_{pridicted}|}" title="https://latex.codecogs.com/svg.image?\large \mathrm{L1\ loss}\mathbf{\ ={\color{Purple} \sum_{i=1}^{n}|y_{true}-y_{pridicted}|}" />

### 2) L2 Loss Function
It is also used to minimize the error which is the sum of all the squared differences in between the true value and the pedicted value.

<img src="https://latex.codecogs.com/svg.image?\large&space;\mathrm{L2\&space;loss}\mathbf{\&space;={\color{Purple}&space;&space;\sum_{i=1}^{n}(y_{true}-y_{pridicted})^2}" title="https://latex.codecogs.com/svg.image?\large \mathrm{L2\ loss}\mathbf{\ ={\color{Purple} \sum_{i=1}^{n}(y_{true}-y_{pridicted})^2}" />

**The disadvantage** of the **L2 norm** is that when there are outliers, these points will account for the main component of the loss. For example, the true value is 1, the prediction is 10 times, the prediction value is 1000 once, and the prediction value of the other times is about 1, obviously the loss value is mainly dominated by 1000.
