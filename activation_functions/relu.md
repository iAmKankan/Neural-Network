## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

## ReLU (_Rectified Linear Unit function_)
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
* The most popular choice, due to both simplicity of implementation and its good performance on a variety of predictive tasks, is the _`rectified linear unit (ReLU)`_.
* ReLU provides a very simple _**nonlinear transformation**_.
> #### Function
*  _g(x) = max(x,0)_
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
* There is an old adage that if subtle boundary conditions matter, we are probably doing (real) mathematics, not engineering. That conventional wisdom may apply here. We plot the derivative of the ReLU function plotted below.
<img src="https://user-images.githubusercontent.com/12748752/146598664-52c52230-8f50-49a4-8e27-e4f35f735726.png" width=40% />

* The reason for using ReLU is that its derivatives are particularly well behaved: either they vanish or they just let the argument through.
* This makes optimization better behaved and it mitigated the well-documented problem of vanishing gradients that plagued previous versions of neural networks.
