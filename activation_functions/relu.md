## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

## ReLU (_Rectified Linear Unit function_)
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
* The most popular choice, due to both simplicity of implementation and its good performance on a variety of predictive tasks, is the _`rectified linear unit (ReLU)`_.
* ReLU provides a very simple _**nonlinear transformation**_.
> #### Function
*  **_g(x) = max(x,0)_**
<img src="https://latex.codecogs.com/svg.image?R(z)=\begin{Bmatrix}z&space;&&space;z&space;>&space;0&space;\\0&space;&&space;z<=0&space;\\\end{Bmatrix}" title="R(z)=\begin{Bmatrix}z & z > 0 \\0 & z<=0 \\\end{Bmatrix}" />

```Python
def relu(z):
  return max(0, z)
```
> #### Given an element  x , the function is defined as the maximum of that element and  0 : _`ReLU(x) = max(x,0)`_ .
* Informally, the ReLU function retains only positive elements and discards all negative elements by setting the corresponding activations to 0. 
* As you can see, the activation function is piecewise linear.
<img src="https://user-images.githubusercontent.com/12748752/146598670-fcbad072-91b1-4a5e-b6c3-33ccc2db8bc0.png" width=40% />

> #### Derivative
<img src="https://latex.codecogs.com/svg.image?R'(z)=\begin{Bmatrix}1&space;&&space;z&space;>&space;0&space;\\0&space;&&space;z<0&space;\\\end{Bmatrix}" title="R'(z)=\begin{Bmatrix}1 & z > 0 \\0 & z<0 \\\end{Bmatrix}" />

```Python
def relu_prime(z):
  return 1 if z > 0 else 0
```
* **When the input is negative,** the derivative of the _ReLU_ function is _**0**_, 
* **When the input is positive,** the derivative of the _ReLU_ function is _**1**_. 
* **Note** that the ReLU function is not _differentiable_ when the input takes value precisely equal to 0. 
* In these cases, we default to the left-hand-side derivative and say that the derivative is 0 when the input is 0.
* We can get away with this because the input may never actually be zero. 
<img src="https://user-images.githubusercontent.com/12748752/146598664-52c52230-8f50-49a4-8e27-e4f35f735726.png" width=40% />

* _ReLU_ behave much better in deep neural networks, mostly because it does not saturate for positive values (and because it is fast to compute).
#### _dying ReLUs_ Problem
* During training, some neurons effectively “die,” meaning they stop outputting anything other than 0. 
* In some cases, you may find that half of your network’s neurons are dead, especially if you used a large learning rate. 
* A neuron dies when its weights get tweaked in such a way that the weighted sum of its inputs are negative for all instances in the training set. 
* When this happens, it just keeps outputting zeros, and Gradient Descent does not affect it anymore because the gradient of the ReLU function is zero when its input is negative.

### _leaky ReLU_
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
* To solve the  _dying ReLUs_ problem, we use leaky ReLU. 
> #### Function 
* _**LeakyReLU (z) = max(αz, z)**_
* The hyperparameter **_α_** defines how much the function “leaks”: it is the slope of the function for z < 0 and is typically set to 0.01. This small slope ensures that leaky ReLUs never die; they can go into a long coma, but they have a chance to eventually wake up. A 2015 paper compared several variants of the ReLU activation function, and one of its conclusions was that the leaky variants always outperformed the strict ReLU activation function. In fact, setting α = 0.2 (a huge leak) seemed to result in better performance than α = 0.01 (a small leak). The paper also evaluated the randomized leaky ReLU (RReLU), where α is picked randomly in a given range during training and is fixed to an average value during testing. RReLU also performed fairly well and seemed to act as a regularizer (reducing the risk of overfitting the training set). Finally, the paper evaluated the parametric leaky ReLU (PReLU), where α is authorized to be learned during training (instead of being a hyperparameter, it becomes a parameter that can be modified by backpropagation like any other parameter). PReLU was reported to strongly outperform ReLU on large image datasets, but on smaller datasets it runs the risk of overfitting the training set.
<img src="https://user-images.githubusercontent.com/12748752/146623578-642cb1b9-d04e-4ede-9b4a-86355049dc23.png" width=40% />
<img src="https://latex.codecogs.com/svg.image?R(z)=\begin{Bmatrix}z&space;&&space;z&space;>&space;0&space;\\\alpha&space;z&space;&&space;z<=0&space;\\\end{Bmatrix}" title="R(z)=\begin{Bmatrix}z & z > 0 \\\alpha z & z<=0 \\\end{Bmatrix}" />

<img src="https://user-images.githubusercontent.com/12748752/146623581-05a0697b-f8fb-4189-bfd7-59fce72cf7dc.png" width=40% />
<img src="https://latex.codecogs.com/svg.image?R'(z)=\begin{Bmatrix}1&space;&&space;z&space;>&space;0&space;\\\alpha&space;&&space;z<0&space;\\\end{Bmatrix}" title="R'(z)=\begin{Bmatrix}1 & z > 0 \\\alpha & z<0 \\\end{Bmatrix}" />


> ## in general SELU > ELU > leaky ReLU (and its variants) > ReLU > tanh > logistic.
