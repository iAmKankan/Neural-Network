## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

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

### _Glorot and He Initialization_
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* In their paper, Glorot and Bengio propose a way to significantly alleviate the unstable gradients problem. 
* They point out that we need the signal to flow properly in both directions: in the forward direction when making predictions, and in the reverse direction.
* We don’t want the signal to die out, nor do we want it to explode and saturate. 
* For the signal to flow properly, the authors argue that we need the variance of the outputs of each layer to be equal to the variance of its inputs, and we need the gradients to have equal variance before and after flowing through a layer in the reverse direction.
* It is actually not possible to guarantee both unless the layer has an equal number of inputs and neurons (these numbers are called the _**fan-in**_ and **_fan-out_** of the layer),
* But they proposed a good compromise that has proven to work very well in practice: **_the connection weights of each layer must be initialized randomly_**
* Where fan<sub>avg</sub> = (_fan<sub>in</sub>_ + _fan<sub>out</sub>_) /2.
* This initialization strategy is called **_`Xavier initialization`_** or _**`Glorot initialization`**_, after the paper’s first author.
> #### _`Glorot initialization`_ (when using the _logistic activation function_)
 <img src="https://latex.codecogs.com/svg.image?\\&space;\textrm{Normal&space;distribution&space;with&space;mean&space;0&space;and&space;variance\&space;\&space;}\sigma^2\&space;=&space;\frac{1}{fan_{avg}}\\&space;\\&space;\textrm{Or&space;a&space;uniform&space;distribution&space;between&space;-r&space;and&space;&plus;r,&space;with&space;\&space;\&space;r\&space;=\&space;}&space;\sqrt{\frac{3}{fan_{avg}}}&space;" title="\\ \textrm{Normal distribution with mean 0 and variance\ \ }\sigma^2\ = \frac{1}{fan_{avg}}\\ \\ \textrm{Or a uniform distribution between -r and +r, with \ \ r\ =\ } \sqrt{\frac{3}{fan_{avg}}} " />

* If we replace _fan<sub>avg</sub>_ with _fan<sub>in</sub>_ , we get ***LeCun initialization*** .
> #### ReLU initialization strategy
* The initialization strategy for the ReLU activation function and its variants, including the ELU activation is sometimes called **_He initialization_**, 
* The SELU activation function will be explained later in this chapter. It should be used with LeCun initialization (preferably with a normal distribution).



> #### **_Initialization parameters for each type of activation function_**
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
<img src="https://latex.codecogs.com/svg.image?\begin{matrix}\mathbf{Initialization}&space;&&space;\textbf{Activation&space;functions}&space;&\sigma^2\textbf{(Normal)}&space;&space;\\&space;\\&space;Glorot&None,\&space;tanh,\&space;logistic,\&space;softmax&space;&space;&&space;\frac{1}{fan_{avg}}&space;\\&space;\\&space;&space;He&&space;ReLU\&space;and\&space;variants&space;&&space;\frac{2}{fan_{in}}&space;\\&space;\\&space;LeCun&&space;SELU&space;&&space;\frac{1}{fan_{in}}\end{matrix}" title="\begin{matrix}\mathbf{Initialization} & \textbf{Activation functions} &\sigma^2\textbf{(Normal)} \\ \\ Glorot&None,\ tanh,\ logistic,\ softmax & \frac{1}{fan_{avg}} \\ \\ He& ReLU\ and\ variants & \frac{2}{fan_{in}} \\ \\ LeCun& SELU & \frac{1}{fan_{in}}\end{matrix}" />

### _Activation function Initialization in Keras_
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* By default, `Keras uses` **_`Glorot initialization`_** with a `uniform distribution`. 
### _`He initialization`_ in Keras
   * `kernel_initializer="he_uniform"` 
   * `kernel_initializer="he_normal"` 
```Python 
keras.layers.Dense(10, activation="relu", kernel_initializer="he_normal") 
````
* _`uniform distribution`_ but based on `fan<sub>avg</sub>` rather than `fan<sub>in</sub>`, then **_`VarianceScaling`_** initializer like this: 
```Python
he_avg_init = keras.initializers.VarianceScaling(scale=2., mode='fan_avg', distribution='uniform') 
keras.layers.Dense(10, activation="sigmoid", kernel_initializer=he_avg_init)
```
