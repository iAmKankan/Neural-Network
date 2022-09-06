## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* [Recurrent Neural Networks(RNN)](#recurrent-neural-networksrnn)
  * [Output of a recurrent layer for a single instance](#output-of-a-recurrent-layer-for-a-single-instance)
  * [Memory Cells](#memory-cells)
  * [Different types of RNN based on Input and Output Sequences](#different-types-of-rnn-based-on-input-and-output-sequences)
    * [Sequence-to-Sequence Network](#sequence-to-sequence-network)
    * [Sequence-to-Vector Network](#sequence-to-vector-network)
    * [Vector-to-Sequence Network](#vector-to-sequence-network)
    * [Encoder-Decoder Network](#encoder-decoder-network)
  * [Backpropagation or Training RNNs](#backpropagation-or-training-rnns)

## Recurrent Neural Network (RNN)
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
A Recurrent Neural Network is a type of ANN that contains **loops**, allowing **information to be stored within the network**. It is absolutely essential for **sequential like information** (**variable input size**)

## Why Recurrent Neural Networks(RNN)?
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

**RNN** works with -
  * **Variable length Input**
  * **Sequential Data**: Text data is a example of sequential data, where the context of previous input matters for the present as well as for the future inputs.


Artificial Neural Network(ANN) and Convolutional Neural Network(CNN) are not capable of handling Sequential data or time series, e.g. **1) Text, 2) Audio, 3) Video**, because they take the followings- 
* Fixed sized inputs
* The whole input available simultaneously

Whereas the RNN takes **variable sized input** inorder to process the sequential information. In the following problems-
* **Speech processing**
* **Language Translation**
* **Video analysis**

Variable sized input where sequential information matters



#### Drawbacks of CNN/ANN
* No memory element. 
* The present data doesn't dependent on the pevious data.


### Different types of RNN based on Input and Output Sequences
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

<p align="center">
<img src="https://user-images.githubusercontent.com/12748752/144064476-e96cb279-6ea7-4a07-aad8-d883b0549bc2.png" width=50% />
<br> <ins><b><i>Many-to-Many (top left), Many-to-One (top right), One-to-Many (bottom left), and Encoder–Decoder (bottom right) networks </i></b></ins>
</p>

### Many-to-Many(Sequence-to-Sequence) Network
* An RNN can simultaneously take a sequence of inputs and produce a sequence of outputs.
* **Example #1:** **Language Translation**, **Speech Recognition**
  * The output does not comes simultaniously with the input and **the size of the output need not to be same as input**
* **Example #2:** Video frame by frame analysis
  * The output size fixed by the input size
   
### Many-to-One (Sequence-to-Vector) Network
* You could feed the network a sequence of inputs and ignore all outputs except for the last one. 
* For example: **Sentiment Analysis**- you could feed the network a sequence of words corresponding to a **movie review** and the network would output a **sentiment score**.

###  One-to-Many(Vector-to-Sequence) Network
* Conversely, you could feed the network the same input vector over and over again at each time step and let it output a sequence.
* For example: **Image Captioning**- the input could be an image (or the output of a CNN), and the output could be a caption(text) for that image.

### Encoder-Decoder Network
* Lastly, you could have a **sequence-to-vector network**, called an **`encoder`**, followed by a **vector-to-sequence network**, called a **`decoder`**. 
* For example, this could be used for translating a sentence from one language to another. 
  * You would feed the network a sentence in one language, the encoder would convert this sentence into a single vector representation, and then the decoder would decode this vector into a sentence in another language. 
  * This two-step model, called an `Encoder–Decoder`, works much better than trying to translate on the fly with a single `sequence-to-sequence RNN` (like the one represented at the top left): the last words of a sentence can affect the first words of the translation, so you need to wait until you have seen the whole sentence before translating it.
  
## Recurrent Neural Networks(RNN)
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

<p align="center">
<img src="https://user-images.githubusercontent.com/12748752/144027506-ea345023-d777-493c-88fd-fae058e70097.png" width=50%/>
<br><ins><b><i>A recurrent neuron (left)    |    unrolled through time (right)</i></b></ins>
</p>

In general- **variably sized**, **sequential data** combine an input vector with a state vector via a fixed function to produce a new state.
* **variably sized**: Number of features are fixed the size of the data is not.
* It looks very much like a feedforward neural network, except it also has connections pointing backward. 

#### Incorporate with RNN the idea of equally spread, repetative, temporal relationship

<img src="https://user-images.githubusercontent.com/12748752/188321976-67e6563e-d6cf-4387-933d-7bda64a88131.png" width=70% />

### Hidden Layers:

### What is the most general function we typically use within neural network? 
**Answer:** We take **linear combination** followed by **non-linearity** always. Typically in **RNNs** we usually use **tanh** for the **nonlinearity** in the **hidden layers**. 

$$\Huge{\color{Purple}
\begin{align*}
& \ h_t = f_w (h_{t-1}, x_t) &\\
& \Huge \boxed{h_t = tanh (W_{{\color{Cyan}hh}}h_{t-1},W_{{\color{Cyan}xh}} x_t) + \textrm{b}}  & \normalsize \begin{cases}
W_{{\color{Cyan}hh}} = \textit{ takes an \textbf{h} and gives out an \textbf{h}} \\
W_{{\color{Cyan}xh}} = \textit{ takes an \textbf{x} and gives out an \textbf{h}}\\ 
W_{{\color{Cyan}xh}}, W_{{\color{Cyan}xh}} , \textbf{b}_n = \textit{ are constant with time}
\end{cases}
\end{align*}
}
$$

* So in this case, the non-linear function is **tanh** and we need a linear combination of **h** and **x**. So there will be some **weight matrix** **_W_** which we will multiply **h** and some other **weight matrix** **_W_** which we will multiply **x**. 
* Those two weight matrices are different in general. Not only that, they also have different sizes.
* So this is the general formula for the hidden layer of an **RNN**, some people will replace this tanh by **_f<sub>w</sub>_** or by **_g_**.

The calculation for hiddenlayer **_h<sub>2</sub>_**  would be-

$$
\large{\color{Purple} h_2 = tanh (W_{hh}h_{1},W_{xh} x_2) + \textrm{b}} 
$$

### Output Layer: 
The Output size is variable. 
What about this $\large{\color{Purple} \hat{y}_t}$ ?

**Answer:**  $\large{\color{Purple} \hat{y}_t}$ is equal to some function of $\large{\color{Purple} h_t}$. 

Now in some cases, it simply makes sense for this function to be a **linear function**( for regression output). In some cases, it makes sense for the function to be a **non-linear function** ( for Classification output).
* If it is a classifcation task and let us say it is a binary classification task, then **g** will become a **Sigmoid** &sigma; . 
* If it is a multiclass classification task, you will use a **Softmax**.

### Constant with time meaning
**Answer:** The **weights** and **bias** in **_h<sub>3</sub>_** are the same **_W<sub>hh</sub>  ,  W<sub>xh</sub>_** and  **_b<sub>n</sub>_** for **_h<sub>2</sub>_**

![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)


### Training RNN
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

#### Calculating Loss in RNN
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

<p align="center"> 
<img src="https://user-images.githubusercontent.com/12748752/188473359-24396c3f-04df-487b-90a6-3a5837be0cf2.png" width=40%/>
<br><ins><b><i> RNN- Many-To-Many  |  The number of Layers = 'T'</i></b></ins>
</p>

#### Example:
* Now when you have **multiple predicted values**, let us say having **10 days** before is the weather of **_h<sub>0</sub>_** or temperature of **_x<sub>0</sub>_** in some city, let us say Chennai. 
* So suppose you have that input, you would have the next day's temperature, let us say that is <img src="https://latex.codecogs.com/svg.image?\large&space;\hat{y}_1" title="https://latex.codecogs.com/svg.image?\large \hat{y}_1" align="center"/>, the next day's temperature <img src="https://latex.codecogs.com/svg.image?\large&space;\hat{y}_2" title="https://latex.codecogs.com/svg.image?\large \hat{y}_2" align="center" />, next day's temperature <img src="https://latex.codecogs.com/svg.image?\large&space;\hat{y}_3" title="https://latex.codecogs.com/svg.image?\large \hat{y}_3" align="center"/>, till let us say today's temperature which is <img src="https://latex.codecogs.com/svg.image?\large&space;\hat{y}_{\textrm{T}}" title="https://latex.codecogs.com/svg.image?\large \hat{y}_{\textrm{T}}" align="center"/> . 
* Now for each one of them, you also have a corresponding ground truth, which should be <img src="https://latex.codecogs.com/svg.image?\large&space;y_1,\&space;y_2,\&space;y_3,\&space;y_T" title="https://latex.codecogs.com/svg.image?\large y_1,\ y_2,\ y_3,\ y_T" align="center"/> . And whenever you have a ground truth and a prediction and these two differs, you will have a **loss function**. 
* So the total loss is -

$$
\Huge{\color{Purple} 
\begin{align*}
& \boxed{ \textbf{L} = \sum_{t=1}^{\textrm{T}} \textbf{L}_{t} } & 
\Big \\{ \normalsize \textit{ Summation of all the intermediate losses through the layers} \\
\end{align*}
}
$$

* Now in terms of **_L<sub>t</sub>_** itself, or the **local loss function**, you again have many choices but we having seen only 2 so far,
  1. [cross entropy](https://github.com/iAmKankan/MachineLearning_With_Python/edit/master/Supervised/Logistic%20Regrassion/README.md#binary-cross-entropy-cost-function) -classifcation
  2. least-squares error - regression or a numerical output

#### Backpropagation Through Time (BPTT)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
#### We know
The following expression for any **h<sub>t</sub>** - 

$$ \large{\color{Purple} h_t = g (W_{hh}h_{t-1}+W_{xh} x_t) + \textrm{b}} $$


 * The **non-linear function** for here is **g** 
 * The **linear combination** of **h** and **x** and **weight matrix** **_W_** which we will multiply **h** of previous layer and some other **weight matrix** **_W_** which we will multiply **x**. 
 * **W<sub>hh</sub>** takes an **h** and gives out an **h**.
 * **W<sub>rh</sub>** takes an **x** and gives out an **h**.

The following expression for any <img src="https://latex.codecogs.com/svg.image?\large&space;\hat{y}_t" title="https://latex.codecogs.com/svg.image?\large \hat{y}_t" align="center"/>, **g** needs not to be same as **g<sup>*</sup>** even **g<sup>*</sup>** not always be **non-linear function**- 

$$ \large{\color{Purple} \hat{y_t} = g^* ( W_{yh} h_t) } $$


#### Inorder to make the expressions simple we put allias in the above two expressionas like 

<img src="https://user-images.githubusercontent.com/12748752/188525079-36af334d-6d36-4550-8480-8094a409168a.png" width= 55%/>

#### **_h<sub>t</sub>_** and <img src="https://latex.codecogs.com/svg.image?\large&space;\hat{y}_t" title="https://latex.codecogs.com/svg.image?\large \hat{y}_t" align="center"/>  looks like- 
$$ \Huge{\color{Purple} \begin{align*}
& h_t = g (W h_{t-1}+ U x_t) + \textrm{b} \\
&  \hat{y_t} = g^* ( V h_t) \\
\end{align*}} $$

#### How does RNN keep the context?

Answer: <img src="https://latex.codecogs.com/svg.image?{\color{Purple}\textrm{W,&space;U,&space;V&space;}&space;}" title="https://latex.codecogs.com/svg.image?{\color{Purple}\textrm{W, U, V } }" align="center" /> do not change with time (or across the layers).


#### For the Backprop we need to findout the followings- 
$$
\Huge \boxed{{\color{Purple} \frac{\partial \textrm{L}}{\partial \textrm{W}},\ \frac{\partial \textrm{L}}{\partial \textrm{U}},\ \frac{\partial \textrm{L}}{\partial \textrm{V}} }}
$$


![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

<img src="https://user-images.githubusercontent.com/12748752/144035005-3e1f7cb9-3cd8-4f2f-9d11-98a4bfc61ce0.png" width=50%/>

> ####  <ins>A recurrent neuron (left)    |    unrolled through time (right)</ins>

* Let’s look at the simplest possible RNN, composed of one neuron receiving inputs, producing an output, and sending that output back to itself. 
* At each time step `t` (also called a frame), this recurrent neuron receives the inputs `x` as well as its own output from the previous time step, y .
* Since there is no previous output at the first time step, it is generally set to 0. 
* We can represent this tiny network against the time axis, as shown above.
* This is called _**`unrolling the network through time`**_ (it’s the same recurrent neuron represented once per time step).

> #### At each time step `t`, every neuron receives both the `input vector` x<sub>(t)</sub> and the `output vector from the previous time step ` y<sub>(t-1)</sub> 
>> #### Note that both the inputs and outputs are vectors now (when there was just a single neuron, the output was a scalar). 

 ### Output of a recurrent layer for a single instance
 ![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

<img src="https://latex.codecogs.com/svg.image?\textbf{y}_{(t)}&space;=&space;\phi&space;(\textbf{W}_x^\top\textbf{x}_{(t)}&plus;\textbf{W}_y^\top\textbf{y}_{(t-1)}&plus;\textbf{b})" title="\textbf{y}_{(t)} = \phi (\textbf{W}_x^\top\textbf{x}_{(t)}+\textbf{W}_y^\top\textbf{y}_{(t-1)}+\textbf{b})" />

* Each recurrent neuron has two sets of weights: one for the inputs <img src="https://latex.codecogs.com/svg.image?\textbf{x}_{(t)}" title="\textbf{x}_{(t)}" /> and the other for the outputs of the previous time step, <img src="https://latex.codecogs.com/svg.image?\textbf{y}_{(t-1)}" title="\textbf{y}_{(t-1)}" height=50%/> .
* Let’s call these weight vectors <img src="https://latex.codecogs.com/svg.image?\textbf{w}_x" title="\textbf{w}_x" /> and <img src="https://latex.codecogs.com/svg.image?\textbf{w}_y" title="\textbf{w}_y" />. 
* If we consider the whole recurrent layer instead of just one recurrent neuron, we can place all the weight vectors in two weight matrices, <img src="https://latex.codecogs.com/svg.image?\textbf{W}_x" title="\textbf{W}_x" /> and <img src="https://latex.codecogs.com/svg.image?\textbf{W}_y" title="\textbf{W}_y" /> .

### Memory Cells
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* Since the output of a recurrent neuron at time step `t` is a function of all the inputs from previous time steps, you could say it has a form of `memory`.
* A part of a neural network that preserves some state across time steps is called a `memory cell` (or simply a `cell`). 
* A single recurrent neuron, or a layer of recurrent neurons, is a very basic cell, capable of learning only short patterns (typically about 10 steps long, but this varies depending on the task).
* In general a cell’s state at time step `t`, denoted <img src="https://latex.codecogs.com/svg.image?\textbf{h}_{(t)}" title="\textbf{h}_{(t)}" /> (the “h” stands for “hidden”), is a function of some inputs at that time step and its state at the previous time step:<img src="https://latex.codecogs.com/svg.image?\textbf{h}_{(t)}=&space;f(\textbf{h}_{(t-1)},\textbf{x}_{(t)})" title="\textbf{h}_{(t)}= f(\textbf{h}_{(t-1)},\textbf{x}_{(t)})" />. 
* Its output at time step `t`, denoted <img src="https://latex.codecogs.com/svg.image?\textbf{y}_{(t)}" title="\textbf{y}_{(t)}" height=50%/> , is also a function of the previous state and the current inputs. 
* In the case of the basic cells we have discussed so far, the output is simply equal to the state, but in more complex cells this is not always the case

<img src="https://user-images.githubusercontent.com/12748752/144058855-ddbd4576-7fbe-4ed0-89c8-3c9ba29655b2.png" width=40%>

### Backpropagation or Training RNNs
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* [Backpropagation in common ANN or Multi-Layer Perceptron](https://github.com/iAmKankan/Neural-Network/blob/main/multilayer-perceptrons.md#backpropagationhttps://github.com/iAmKankan/Neural-Network/blob/main/multilayer-perceptrons.md#backpropagation)
* To train an RNN, the trick is to unroll it through time (like we just did) and then simply use regular backpropagation. 
* This strategy is called ***`backpropagation through time (BPTT)`***.
<img src="https://user-images.githubusercontent.com/12748752/144243558-a7cae1ca-96d7-4d80-9be8-bb4e7e960dc4.png" width=50%/>

#### First
* Like regular backpropagation, there is a `first forward pass through the unrolled network` (represented by the dashed arrows). 
#### Then 
* The output sequence is evaluated using a cost function <img src="https://latex.codecogs.com/svg.image?C(\textbf{Y}_{(0)},&space;\textbf{Y}_{(1)},...&space;,&space;\textbf{Y}_{(T)})" title="C(\textbf{Y}_{(0)}, \textbf{Y}_{(1)},... , \textbf{Y}_{(T)})" /> (where _T_ is the max time step). 
* **Note**: this cost function may `ignore some outputs`(for example, in a sequence-to-vector RNN, all outputs are ignored except for the very last one). 
#### Then
* The gradients of that cost function are then `propagated backward through the unrolled network` (represented by the solid arrows). 
#### Finally 
* The `model parameters are updated` using the gradients computed during **BPTT**. 
* **Note** that the gradients flow backward through all the outputs used by the cost function, not just through the final output (for example, in Figure the cost function is computed using the last three outputs of the network, <img src="https://latex.codecogs.com/svg.image?\textbf{Y}_{(2)},&space;\textbf{Y}_{(3)}\&space;and&space;\&space;\textbf{Y}_{(4)}" title="\textbf{Y}_{(2)}, \textbf{Y}_{(3)}\ and \ \textbf{Y}_{(4)}" />, so gradients flow through these three outputs, but not through **Y<sub>(0)</sub>** and **Y<sub>(1)</sub>** ). 
* Moreover, since the same parameters **W** and **b** are used at each time step, backpropagation will do the right thing and sum over all time steps.
* Fortunately, tf.keras takes care of all of this complexity for you

## Problems in Training Simple RNNs
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
1. Unstable Gradient

### Rolled RNN and Unrolled version of RNN

## Bibliography
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition by Aurélien Géron**
