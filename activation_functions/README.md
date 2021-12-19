## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* [Activation Function](#activation-function)
* [Why do we need activation functions in the first place](#why-do-we-need-activation-functions-in-the-first-place)
* [Different types of Activation functions and their derivatives](#different-types-of-activation-functions-and-their-derivatives)
* [How to use Activation functions in Keras](#how-to-use-activation-functions-in-keras)
   * [Usage of activations](#usage-of-activations)
* [The Vanishing or Exploding Gradients Problems](#the-vanishing-or-exploding-gradients-problems)
   * [Backpropagation](#backpropagation-link-for-backpropagation-in-general)
   * [Vanishing Gradients](#vanishing-gradients)
   * [Exploding Gradients](#exploding-gradients)
   * [Problem with Logistic or Sigmoid Activation Function](#problem-with-logistic-or-sigmoid-activation-function)
* [Glorot and He Initialization](#glorot-and-he-initialization)
   * [Glorot initialization (when using the logistic activation function)](#glorot-initialization-when-using-the-logistic-activation-function)
   * [ReLU initialization strategy](#relu-initialization-strategy)
   * [Initialization parameters for each type of activation function](#initialization-parameters-for-each-type-of-activation-function)
   * [Activation function Initialization in Keras](#activation-function-initialization-in-keras)
   * [He initialization in Keras](#he-initialization-in-keras)
* [Activation Function](#activation-function)
   * [Sigmoid](https://github.com/iAmKankan/Neural-Network/blob/main/activation_functions/sigmoid.md)
   * [Softmax](https://github.com/iAmKankan/Neural-Network/blob/main/activation_functions/softmax.md)
   * [TanH](#tanh)
* [Bibliography](#bibliography)
## Activation Function
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* **_Activation functions_** decide whether a neuron should be activated or not by calculating the **_weighted sum_** and further **_adding bias_** with it. 
* They are differentiable operators to transform input signals to outputs, while most of them add non-linearity.

### Why do we need activation functions in the first place
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
* If you chain several linear transformations, all you get is a linear transformation.
> **For example: Say we have f(x) and g(x) then Then chaining these two linear functions gives you another linear function f(g(x)).**
>> f(x) = 2 x + 3 
>
>> g(x) = 5 x - 1 
>
>> f(g(x)) = 2(5 x - 1) + 3 = 10 x + 1.


* So, if you don’t have some non-linearity between layers, then even a deep stack of layers is equivalent to a single layer.
* You cannot solve very complex problems with that.

> ### The botton line is _linear activation function_ cannot be used in _hidden layers_, it has to be at the end if there is a requirment i.e for _regression output layer_ for some special cases
### Different types of Activation functions and their derivatives
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)

<img src="https://user-images.githubusercontent.com/12748752/146569902-e5b03528-bb1b-4a96-a8c6-cf2e998dd0c6.png"/>

### Performance of Different types of Activation functions
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
<img src="https://user-images.githubusercontent.com/12748752/146654612-4d383821-c4b0-46b3-b380-bde4172a9264.png" width=110% />

### How to use Activation functions in _`Keras`_
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
> ### Usage of activations
* Activations can either be used through an _`Activation`_ layer, or through the _`activation`_ argument supported by all forward layers:
```Python
model.add(layers.Dense(64, activation=activations.relu))
```
* Same like the following
```Python
from tensorflow.keras import layers
from tensorflow.keras import activations

model.add(layers.Dense(64))
model.add(layers.Activation(activations.relu))
```
* or String identifier:
```Python
model.add(layers.Dense(64, activation='relu'))
```
## The Vanishing or Exploding Gradients Problems
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
### Backpropagation: [_`link for backpropagation in general`_](https://github.com/iAmKankan/Neural-Network/blob/main/multilayer-perceptrons.md#backpropagation)
* The backpropagation algorithm works by going from the output layer to the input layer, propagating the error gradient along the way. 
* Once the algorithm has computed the gradient of the cost function with regard to each parameter in the network, it uses these gradients to update each parameter with a Gradient Descent step.

### _Vanishing Gradients_
* During Backpropagation gradients often get smaller and smaller as the algorithm progresses down to the lower layers.
* As a result, the Gradient Descent update leaves the lower layers’ connection weights virtually unchanged, and training never converges to a good solution. We call this the vanishing gradients problem.

### _Exploding Gradients_
* In some cases, the opposite of vanishing gradients can happen: the gradients can grow bigger and bigger until layers get insanely large weight updates and the algorithm diverges.
* More generally, deep neural networks suffer from unstable gradients; different layers may learn at widely different speeds.

### _Problem with Logistic or Sigmoid Activation Function_
* In logistic activation function,when inputs become large (negative or positive), the function saturates at 0 or 1, with a derivative extremely close to 0. 
* Thus, when backpropagation kicks in it has virtually no gradient to propagate back through the network; and what little gradient exists keeps getting diluted as backpropagation progresses down through the top layers, so there is really nothing left for the lower layers.
<img src="https://user-images.githubusercontent.com/12748752/146207390-7b35242a-d980-4a6d-8db5-60faba8a5406.png" width=40% />

## _Glorot and He Initialization_
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* In their paper, Glorot and Bengio propose a way to significantly alleviate the unstable gradients problem. 
* They point out that we need the signal to flow properly in both directions: in the forward direction when making predictions, and in the reverse direction.
* We don’t want the signal to die out, nor do we want it to explode and saturate. 
* For the signal to flow properly, the authors argue that we need the variance of the outputs of each layer to be equal to the variance of its inputs, and we need the gradients to have equal variance before and after flowing through a layer in the reverse direction.
* It is actually not possible to guarantee both unless the layer has an equal number of inputs and neurons (these numbers are called the _**fan-in**_ and **_fan-out_** of the layer),
* But they proposed a good compromise that has proven to work very well in practice: **_the connection weights of each layer must be initialized randomly_**
* Where fan<sub>avg</sub> = (_fan<sub>in</sub>_ + _fan<sub>out</sub>_) /2.
* This initialization strategy is called **_`Xavier initialization`_** or _**`Glorot initialization`**_, after the paper’s first author.
> ### _`Glorot initialization`_ (when using the _logistic activation function_)
 <img src="https://latex.codecogs.com/svg.image?\\&space;\textrm{Normal&space;distribution&space;with&space;mean&space;0&space;and&space;variance\&space;\&space;}\sigma^2\&space;=&space;\frac{1}{fan_{avg}}\\&space;\\&space;\textrm{Or&space;a&space;uniform&space;distribution&space;between&space;-r&space;and&space;&plus;r,&space;with&space;\&space;\&space;r\&space;=\&space;}&space;\sqrt{\frac{3}{fan_{avg}}}&space;" title="\\ \textrm{Normal distribution with mean 0 and variance\ \ }\sigma^2\ = \frac{1}{fan_{avg}}\\ \\ \textrm{Or a uniform distribution between -r and +r, with \ \ r\ =\ } \sqrt{\frac{3}{fan_{avg}}} " />

* If we replace _fan<sub>avg</sub>_ with _fan<sub>in</sub>_ , we get ***LeCun initialization*** .
> ### ReLU initialization strategy
* The initialization strategy for the ReLU activation function and its variants, including the ELU activation is sometimes called **_He initialization_**, 
* The SELU activation function will be explained later in this chapter. It should be used with LeCun initialization (preferably with a normal distribution).



### **_Initialization parameters for each type of activation function_**
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)

 <img src="https://latex.codecogs.com/svg.image?\begin{matrix}\mathbf{Initialization}&space;&&space;\textbf{Activation&space;functions}&space;&\sigma^2\textbf{(Normal)}&space;&space;\\&space;\\&space;Glorot&None,\&space;tanh,\&space;logistic,\&space;softmax&space;&space;&&space;\frac{1}{fan_{avg}}&space;\\&space;\\&space;&space;He&&space;ReLU\&space;and\&space;variants&space;&&space;\frac{2}{fan_{in}}&space;\\&space;\\&space;LeCun&&space;SELU&space;&&space;\frac{1}{fan_{in}}\end{matrix}" title="\begin{matrix}\mathbf{Initialization} & \textbf{Activation functions} &\sigma^2\textbf{(Normal)} \\ \\ Glorot&None,\ tanh,\ logistic,\ softmax & \frac{1}{fan_{avg}} \\ \\ He& ReLU\ and\ variants & \frac{2}{fan_{in}} \\ \\ LeCun& SELU & \frac{1}{fan_{in}}\end{matrix}" />

> ### _Activation function Initialization in Keras_
* By default, `Keras uses` **_`Glorot initialization`_** with a `uniform distribution`. 

> ### _`He initialization`_ in Keras
* `kernel_initializer="he_uniform"` 
* `kernel_initializer="he_normal"` 
```Python 
keras.layers.Dense(10, activation="relu", kernel_initializer="he_normal") 
````
* _`uniform distribution`_ but based on fan<sub>avg</sub> rather than fan<sub>in</sub>, then **_`VarianceScaling`_** initializer like this: 
```Python
he_avg_init = keras.initializers.VarianceScaling(scale=2., mode='fan_avg', distribution='uniform') 
keras.layers.Dense(10, activation="sigmoid", kernel_initializer=he_avg_init)
```

## Nonsaturating Activation Functions
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* Until then most people had assumed that if Mother Nature had chosen to use roughly sigmoid activation functions in biological neurons, they must be an excellent choice.
* But it turns out that other activation functions behave much better in deep neural networks—in particular, the ReLU activation function, mostly because it does not saturate for positive values (and because it is fast to compute).

## More Links
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* [Keras Doc](https://keras.io/api/layers/activations/#selu-function)

## Bibliography
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

* [Deep in to Deep Learning]()
* 

