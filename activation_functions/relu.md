## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

## ReLU (_Rectified Linear Unit_) function
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* The most popular choice, due to both simplicity of implementation and its good performance on a variety of predictive tasks, is the _`rectified linear unit (ReLU)`_.
* ReLU provides a very simple _**nonlinear transformation**_.
### Function: $\Large{\color{Purple} g(x) = max(x,0)}$

> #### Given an element  x , the function is defined as the maximum of that element and  0 : _`ReLU(x) = max(x,0)`_ .
* Informally, the ReLU function retains only positive elements and discards all negative elements by setting the corresponding activations to 0. 
* As you can see, the activation function is piecewise linear.

<p align="center">
	<img src="https://user-images.githubusercontent.com/12748752/215332857-893ba6cd-0051-4d35-9813-2acd7318c171.png" />
</p>

> #### Relu function
```Python
def relu(z):
  return max(0, z)
```

> #### Derivative

```Python
def relu_prime(z):
  return 1 if z > 0 else 0
```
* **When the input is negative,** the derivative of the _ReLU_ function is _**0**_, 
* **When the input is positive,** the derivative of the _ReLU_ function is _**1**_. 
* **Note** that the ReLU function is not _differentiable_ when the input takes value precisely equal to 0. 
* In these cases, we default to the left-hand-side derivative and say that the derivative is 0 when the input is 0.
* We can get away with this because the input may never actually be zero. 
* _ReLU_ behave much better in deep neural networks, mostly because it does not saturate for positive values (and because it is fast to compute).

### _dying ReLUs_ Problem
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
1. During training, some **neurons** only gives **0** as **output**. This neuron “**die**” scenario is called   **Dying Relu**.
	* In some cases especially if you are using a **large learning rate**, you may find **half of your network’s neurons** are **dead**.

2. **A neuron dies** when its <ins><b>weights get tweaked</b></ins> in such a way that **_the weighted sum of its inputs are negative for all instances in the training set_**. 
	* When this happens, it just keep giving the output zeros, and **Gradient Descent** does not affect it anymore because the gradient of the **ReLU** function is **zero** when its **input is negative**.

## _Leaky ReLU_
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

To solve the **_dying ReLUs_** problem, we use **leaky ReLU**. 

It give you **a small gradient value** for **negative values** of the **input**. 
* So in the plain **ReLu**, if the **input** was **negative** than your **gradient** also go **0**, but this case all **negative values** of **x**, it will give you a **scaled value of that input**.
* Which means that the **gradient** can exist okay. 
	
In other version of this is the Parametric Rectified Linear unit, wherein instead of having a fixed scaling factor for x, for negative values of x, we have an alpha which is again a parameter which is learned during the back propagation process, okay.

###  Function $\Large{\color{Purple} LickeyReLU (z) = max(0.01z, z)}$
###  Function $\Large{\color{Purple} ParametricReLU (z) = max(\alpha z, z)}$

<p align="center">
	<img src="https://user-images.githubusercontent.com/12748752/215331150-91106642-3834-48d4-b092-d206dfdf6be0.png" width=70% /> 
</p>

> #### ReLu code
```Python
def leakyrelu(z, alpha):
	return max(alpha * z, z)
```

> #### Derivative

```Python
def leakyrelu_prime(z, alpha):
	return 1 if z > 0 else alpha
```

* The hyperparameter **_α_** defines how much the function “leaks”: it is the slope of the function for _z < 0_ and is typically set to _0.01_. 
* This small slope ensures that leaky ReLUs never die; they can go into a long coma, but they have a chance to eventually wake up. 
* A 2015 paper compared several variants of the ReLU activation function, and one of its conclusions was that the leaky variants always outperformed the strict ReLU activation function. 
* In fact, setting α = 0.2 (a huge leak) seemed to result in better performance than _`α = 0.01`_ (a small leak). 
#### _**Randomized leaky ReLU (RReLU)**_, 
* Where _**α**_ is picked randomly in a given range during training and is fixed to an average value during testing. 
* RReLU also performed fairly well and seemed to act as a regularizer (reducing the risk of overfitting the training set). 
#### _**Parametric leaky ReLU (PReLU)**_
* Finally, the paper evaluated the _**parametric leaky ReLU (PReLU)**_, where _**α**_ is authorized to be learned during training (instead of being a hyperparameter, it becomes a parameter that can be modified by backpropagation like any other parameter). 
* _**PReLU**_ was reported to strongly outperform ReLU on large image datasets, but on smaller datasets it runs the risk of overfitting the training set.

![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)

