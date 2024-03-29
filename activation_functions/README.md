## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* [Activation Function](#activation-function)
* [Why do we need activation functions in the first place](#why-do-we-need-activation-functions-in-the-first-place)
* [Different types of Activation functions and their derivatives](#different-types-of-activation-functions-and-their-derivatives)
* [How to use Activation functions in Keras](#how-to-use-activation-functions-in-keras)
   * [Usage of activations](#usage-of-activations)
* [The Vanishing or Exploding Gradients Problems](#the-vanishing-and-exploding-gradients-problems)
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
   * [TanH](https://github.com/iAmKankan/Neural-Network/blob/main/activation_functions/tanh.md)
   * [ReLU](https://github.com/iAmKankan/Neural-Network/blob/main/activation_functions/relu.md)
* [Bibliography](#bibliography)
* [Todo 01](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html#vanishing-and-exploding-gradients)


## The Vanishing and Exploding Gradients Problems
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
The question should be asked that why we need so namy types of Activation function! where as at the beggining people used the **step function** and later **sigmoid** function as the  Activation function. 
### _Vanishing Gradients_ ([the RNN version ↗️](https://github.com/iAmKankan/Neural-Network/tree/main/rnn/README.md#vanishing-gradients-and-tbptt))
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)

$$\Large{\color{Purple}\parallel \frac{\partial L }{\partial W}\parallel \to 0} $$

* During [Backpropagation](https://github.com/iAmKankan/Neural-Network/blob/main/backpropagation/README.md#backpropagation) gradients often get smaller and smaller as the algorithm progresses down to the lower layers.
* As a result, the Gradient Descent update leaves the lower layers’ connection weights virtually unchanged, and training never converges to a good solution. We call this the vanishing gradients problem.

#### [⚛️ Why the small number is a problem? ↗️](https://github.com/iAmKankan/Neural-Network/tree/main/rnn/README.md#%EF%B8%8F-why-the-small-number-is-a-problem)

#### Example 01:
* Suppose we have **Sigmoid** as activation function and we use it throughout a very long neural network.
<img src="https://user-images.githubusercontent.com/12748752/167541362-d5995768-7693-4e85-a79b-9fb63419e21a.png" width=70%/>

#### For backpropagation the weight update formula is - 
<img src="https://latex.codecogs.com/svg.image?\large&space;\mathbf{{\color{Purple}&space;W_{1\&space;new}&space;=&space;W_{1&space;\&space;old}-\eta&space;&space;\frac{\partial&space;L&space;}{\partial&space;W_{1\&space;old}}}" title="https://latex.codecogs.com/svg.image?\large \mathbf{{\color{Purple} W_{1\ new} = W_{1 \ old}-\eta \frac{\partial L }{\partial W_{1\ old}}}" align="center"/>

#### When we calculate its gradients we need its darivative as well-
<img src="https://latex.codecogs.com/svg.image?\mathbf{\frac{\partial&space;L&space;}{\partial&space;W_{1&space;\&space;old}}&space;=\&space;{\color{Purple}&space;&space;\frac{\partial&space;L&space;}{\partial&space;O_{51}}&space;*&space;\frac{\partial&space;O_{51}&space;}{\partial&space;O_{41}}*&space;\frac{\partial&space;O_{41}&space;}{\partial&space;O_{31}}*&space;\frac{\partial&space;O_{31}&space;}{\partial&space;O_{21}}*&space;\frac{\partial&space;O_{21}&space;}{\partial&space;W_{1}}" title="https://latex.codecogs.com/svg.image?\mathbf{\frac{\partial L }{\partial W_{1 \ old}} =\ {\color{Purple} \frac{\partial L }{\partial O_{51}} * \frac{\partial O_{51} }{\partial O_{41}}* \frac{\partial O_{41} }{\partial O_{31}}* \frac{\partial O_{31} }{\partial O_{21}}* \frac{\partial O_{21} }{\partial W_{1}}" align="center" />

#### As we know for Sigmoid activation function its Range is between [_0 to 1_], thresold is _0.5_ and its derivative is  [_0 to 0.25_]
* So, in our example when we want to calculate the derivative of weights inorder to update the new weights in the network we get lesser value each time.
* <img src="https://latex.codecogs.com/svg.image?\mathbf{\frac{\partial&space;L&space;}{\partial&space;W_{1&space;\&space;old}}&space;=" title="https://latex.codecogs.com/svg.image?\mathbf{\frac{\partial L }{\partial W_{1 \ old}} =" align="center" /> <img src="https://latex.codecogs.com/svg.image?\textit{\textbf{{\color{Purple}&space;&space;0.25&space;*&space;0.15&space;*&space;0.10&space;*&space;0.05&space;*&space;0.02}}}" title="https://latex.codecogs.com/svg.image?\textit{\textbf{{\color{Purple} 0.25 * 0.15 * 0.10 * 0.05 * 0.02}}}" align="center" />

* And after some point of time we will see that the gradient is no longer updating <img src="https://latex.codecogs.com/svg.image?\large&space;\mathbf{{\color{Purple}&space;W_{(new)}&space;\approx&space;W_{(old)}}&space;}" title="https://latex.codecogs.com/svg.image?\large \mathbf{{\color{Purple} W_{(new)} \approx W_{(old)}} }" align="center"/>
#### This scenario is called _Vanishing Gradients Problem_
> #### To overcome this problem we use advanced activation functions like Relu etc.

### _Exploding Gradients_ ([the RNN version ↗️](https://github.com/iAmKankan/Neural-Network/tree/main/rnn/README.md#vanishing-gradients-and-tbptt))
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)

$$\Large{\color{Purple}\parallel \frac{\partial L }{\partial W}\parallel \to \infty} $$

* In some cases, the opposite of vanishing gradients can happen: the gradients can grow bigger and bigger until layers get insanely large weight updates and the algorithm diverges.
* More generally, deep neural networks suffer from unstable gradients; different layers may learn at widely different speeds.
> #### This problem associates with _weights_, sometimes the weights get big and when it multiplies with the derivative of the ***Activation Function*** it get bigger.. Not necessary the presents of _Sigmoid_ function.

#### Gredient clipping: [↗️](https://cnvrg.io/gradient-clipping/)
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
$\underline{\textbf{Gredient clipping:}}$  This is sort of a **numerical hack**. It is very simple, we decide on a **maximum allowable gradient size**.

* Deside on a maximum allowable Gradient [value/size](https://github.com/iAmKankan/Neural-Network/edit/main/activation_functions/README.md#gradient-is-a-vector)
   * Say $\Large{\color{Purple} \frac{\partial L}{\partial W} = \vec{g}}$. Then we do $\Large{\color{Purple} max \parallel \vec{g} \parallel = G_{max}}$

* Geadient Descent 
   * Calculate $\Large{\color{Purple} \vec{g}}$
   * If **norm** of **g** $\Large{\color{Purple} \parallel \vec{g} \parallel < G_{max}}$ then proceed as usual.
   * If not then we have to redefine a **new gradient** which is equal to the **old gredient** $\Large{\color{Purple} \vec{\dot{g}}= \frac{\vec{g}}{\parallel\vec{g} \parallel} G_{max} }$
##### Keras Code: 
Applying **gradient clipping** in **TensorFlow** models is quite straightforward. 
* The only thing you need to do is pass the **parameter** to the **optimizer function**. 
* All optimizers have a **_`clipnorm`_** and a **_`clipvalue`_** parameters that can be used to **clip** the **gradients**.

Let’s look at **clipping the gradients** using the **`clipnorm`** parameter using the common **MNIST** example.

<p align="center">
  <img src="https://user-images.githubusercontent.com/12748752/215345534-c245bc58-a715-406d-b9dc-c58e414eb66b.jpeg" width=40%/>
  <br>
  <ins>Gradients less than <b>-0.5</b> will be capped to <b>-0.5</b>, and gradients above <b>0.5</b> will be capped to <b>0.5</b>. </ins>
</p>

```Python
import tensorflow as tf
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255., x_test / 255.

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(clipvalue=0.5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
```

The **`clipnorm`** gradient clipping can be applied similarly. In this case, 1 is specified.

```Python
model.compile(
    optimizer=tf.keras.optimizers.Adam(clipnorm=1.0),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
```

#### Gradient is a Vector:
What do I mean by value of **gradient**? Again remember, **gradient is a vector**, so **you cannot give it a value**, 

* you can however give a value to **norm** of **gradient**. So let us say we are dealing with $\frac{\partial L}{\partial W}$ , let me call it **g** vector. So I will say that maximum value allowable of g vector is some G_max. You will decide it, okay, you will decide on what you are comfortable with. Just like our **cut-off** **criterion**, this is an **arbitrary criterion set by you**, it is sort of an engineering solution to the problem, okay.

#### [⚛️ Why the big number is a problem _since &infin; means a big number_? ↗️](https://github.com/iAmKankan/Neural-Network/tree/main/rnn/README.md#%EF%B8%8F-why-the-big-number-is-a-problem-since--means-a-big-number)

![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

<p align="center">
  <img src="https://user-images.githubusercontent.com/12748752/167583923-c90411f4-8b57-4663-93fa-be18ca47a7cb.png" width=45% /> <img src="https://user-images.githubusercontent.com/12748752/167583928-bfe56750-9917-4209-9eb5-04fc81743647.png" width=45% /> 
  <br>
  <img src="https://user-images.githubusercontent.com/12748752/167583932-9d195974-375c-48a6-84db-04fdbcb5fb6b.png" width=45% />
</p>




## Activation Function
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
**_Activation functions_** decide whether a neuron should be activated or not by calculating the **_weighted sum_** and further **_adding bias_** with it. They are differentiable operators to _transform input signals to outputs_, while most of them _add non-linearity_.

Activation functions are a choice that you must make for each layer.  Generally, you can follow this guideline:
1) <ins>_Hidden Layers_ </ins>- **_RELU_**
2) <ins>_Output Layer_ </ins>- **_Softmax_** for classification, **_linear_** for regression.


Some of the common activation functions in Keras are listed here:

* **softmax** - Used for multi-class classification.  Ensures all output neurons behave as probabilities and sum to 1.0.
* **elu** - Exponential linear unit.  Exponential Linear Unit or its widely known name ELU is a function that tend to converge cost to zero faster and produce more accurate results.  Can produce negative outputs.
* **selu** - Scaled Exponential Linear Unit (SELU), essentially **elu** multiplied by a scaling constant.
* **softplus** - Softplus activation function. **_log(exp(x) + 1)_** [Introduced](https://papers.nips.cc/paper/1920-incorporating-second-order-functional-knowledge-for-better-option-pricing.pdf) in 2001.
* **softsign** Softsign activation function. **_x / (abs(x) + 1)_** Similar to tanh, but not widely used.
* **relu** - Very popular neural network activation function.  Used for hidden layers, cannot output negative values.  No trainable parameters.
* **tanh** Classic neural network activation function, though often replaced by relu family on modern networks.
* **sigmoid** - Classic neural network activation.  Often used on output layer of a binary classifier.
* **hard_sigmoid** - Less computationally expensive variant of sigmoid.
* **exponential** - Exponential (base e) activation function.
* **linear** - Pass through activation function. Usually used on the output layer of a regression neural network.

For more information about Keras activation functions refer to the following:

* [Keras Activation Functions](https://keras.io/activations/)
* [Activation Function Cheat Sheets](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html)



### Advanced Activation Functions
Hyperparameters are not changed when the neural network trains. You, the network designer, must define the hyperparameters.  The neural network learns regular parameters during neural network training.  Neural network weights are the most common type of regular parameter.  The "[advanced activation functions](https://keras.io/layers/advanced-activations/)," as Keras call them, also contain parameters that the network will learn during training.  These activation functions may give you better performance than RELU.

* **LeakyReLU** - Leaky version of a Rectified Linear Unit. It allows a small gradient when the unit is not active, controlled by alpha hyperparameter.
* **PReLU** - Parametric Rectified Linear Unit, learns the alpha hyperparameter. 


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

<img src="https://user-images.githubusercontent.com/12748752/146569902-e5b03528-bb1b-4a96-a8c6-cf2e998dd0c6.png" width=60%/>

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

 <img src="https://latex.codecogs.com/svg.image?\begin{matrix}\mathbf{Initialization}&space;&&space;\textbf{Activation&space;functions}&space;&\sigma^2\textbf{(Normal)}&space;&space;\\&space;\\&space;Glorot&None,\&space;tanh,\&space;logistic,\&space;softmax&space;&space;&&space;\frac{1}{fan_{avg}}&space;\\&space;\\&space;&space;He&&space;ReLU\&space;and\&space;variants&space;&&space;\frac{2}{fan_{in}}&space;\\&space;\\&space;LeCun&&space;SELU&space;&&space;\frac{1}{fan_{in}}\end{matrix}" title="\begin{matrix}
  \mathbf{Initialization} & \textbf{Activation functions} &\sigma^2\textbf{(Normal)} \\ \\ Glorot&None,\ tanh,\ logistic,\ softmax & \frac{1}{fan_{avg}} \\ \\ He& ReLU\ and\ variants & \frac{2}{fan_{in}} \\ \\ LeCun& SELU & \frac{1}{fan_{in}}\end{matrix}" />

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
* [Jeff Heaton]()
