## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
### _Recurrent Neural Network (RNN)_
* [What is RNN](#recurrent-neural-networksrnn)
* [Why RNN](#why-recurrent-neural-networksrnn)
  * [Output of a recurrent layer for a single instance](#output-of-a-recurrent-layer-for-a-single-instance)
  * [Memory Cells](#memory-cells)
  * [Different types of RNN based on Input and Output Sequences](#different-types-of-rnn-based-on-input-and-output-sequences)
    * [Sequence-to-Sequence Network](#sequence-to-sequence-network)
    * [Sequence-to-Vector Network](#sequence-to-vector-network)
    * [Vector-to-Sequence Network](#vector-to-sequence-network)
    * [Encoder-Decoder Network](#encoder-decoder-network)
  * [Backpropagation or Training RNNs](#backpropagation-or-training-rnns)

### [Problems of RNN](#problems-in-training-simple-rnns)
* [Vanishing Gradients and TBPTT](#vanishing-gradients-and-tbptt)
  * [Why the big number is a problem since <b>&infin;</b> means a big number?](#%EF%B8%8F-why-the-big-number-is-a-problem-since--means-a-big-number) $\large{\color{Purple}( \infty )}$
  * [Why the small number is a problem?](#%EF%B8%8F-why-the-small-number-is-a-problem)
  * [Gradient clipping for exploding gradients](#%EF%B8%8F-gradient-clipping-for-exploding-gradients)
* [Solution for Vanishing gradients](#%EF%B8%8F-solution-for-vanishing-gradients)
* [Solution for expensive gradient Computation Truncated Back Propagation Through Time(TBPTT)](#%EF%B8%8F-solution-for-expensive-gradient-computation-truncated-back-propagation-through-timetbptt)

### [Deep Recurrent Neural Network (RNN)](#deep-rnns)  

###  _Varients of Recurrent Neural Network_
* [Gated Recurrent Unit (GRU)](https://github.com/iAmKankan/Neural-Network/blob/main/rnn/gru.md)
* [LSTM](https://github.com/iAmKankan/Neural-Network/blob/main/rnn/lstm.md)

## Recurrent Neural Network (RNN)
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
A Recurrent Neural Network is a type of ANN that contains **loops**, allowing **information to be stored within the network**. It is absolutely essential for **sequential like information** (**variable input size**)

## Why Recurrent Neural Networks(RNN)?
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

$$
{\color{Purple} 
\boxed{
\large\begin{align*}
& \textbf{RNN} && \textbf{ANN and CNN}\\
& \textrm{Variable length Input.}&& \textrm{Fixed sized inputs.}\\
& \textrm{Sequential Data or time series data.}&& \textrm{The whole input available simultaneously.}\\
& \textrm{E.g. Text data, where context matters.}&& \\
& \textrm{Used in- Speech processing,}&& \textrm{Used in- other deep learning work.}\\
& \textrm{Language Translation, video analysis.}&& \\
\end{align*}
}
}
$$


#### Drawbacks of CNN/ANN
* **No memory element**. 
* **The present data doesn't dependent on the pevious data**.



  
### RNN Layers
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
In general- **variably sized**, **sequential data** combine an input vector with a state vector via a fixed function to produce a new state.
* **variably sized**: Number of features are fixed the size of the data is not.
* It looks very much like a feedforward neural network, except it also has connections pointing backward. 

#### Incorporate with RNN the idea of equally spread, repetative, temporal relationship

<p align="center">
 
 <img src="https://user-images.githubusercontent.com/12748752/188321976-67e6563e-d6cf-4387-933d-7bda64a88131.png" width=70% />
 <br><ins><i><b>RNN unrolling over time</b></i></ins>
                                                                                                                                 
</p>
                                                                                                                                 
### Hidden Layers:

### ⚛️ What is the most general function we typically use within neural network? 
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

### ⚛️ What about this $\large{\color{Purple} \hat{y}_t}$ ?
**Answer:**  $\large{\color{Purple} \hat{y}_t}$ is equal to some function of $\large{\color{Purple} h_t}$. 

Now in some cases, it simply makes sense for this function to be a **linear function**( for regression output). In some cases, it makes sense for the function to be a **non-linear function** ( for Classification output).
* If it is a classifcation task and let us say it is a binary classification task, then **g** will become a **Sigmoid** &sigma; . 
* If it is a multiclass classification task, you will use a **Softmax**.

### ⚛️ Constant with time meaning
**Answer:** The **weights** and **bias** in **_h<sub>3</sub>_** are the same **_W<sub>hh</sub>  ,  W<sub>xh</sub>_** and  **_b<sub>n</sub>_** for **_h<sub>2</sub>_**


### ⚛️ Calculating Loss in RNN

<p align="center"> 
<img src="https://user-images.githubusercontent.com/12748752/188473359-24396c3f-04df-487b-90a6-3a5837be0cf2.png" width=40%/>
<br><ins><b><i> RNN- Many-To-Many  |  The number of Layers = 'T'</i></b></ins>
</p>

#### Example:
* Now when you have **multiple predicted values**, let us say having **10 days** before is the weather of $\large{\color{Purple} h_0}$ or temperature of $\large{\color{Purple} {x}_0}$ in some city, let us say Chennai. 
* Suppose you have that input, you would have the next day's temperature, let us say that is $\large{\color{Purple} \hat{y_1}}$ , the next day's temperature $\large{\color{Purple} \hat{y_2}}$ and next day's temperature $\large{\color{Purple} \hat{y_3}}$ , till let us say today's temperature which is $\large{\color{Purple} \hat{y_T}}$ .
* Now for each one of them, you also have a corresponding ground truth, which should be $\large{\color{Purple} y_1,\ y_2,\ y_3, \ y_T }$ . And whenever you have a ground truth and a prediction and these two differs, you will have a **loss function**. 
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

## Backpropagation Through Time (BPTT)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

<p align="center"> 
 <img src="https://user-images.githubusercontent.com/12748752/211137192-353752b1-36e1-4cbd-b303-637ba8e88dd3.png" width=60% />
 <br><ins><b><i> Weight matrix for a single hidden layer RNN | RNN- Many-To-Many  |  Total number of Layers = 'T'</i></b></ins>
</p>


### For _Hiddden layers_ for any $\large{\color{Purple} h_t}$ - 

$$ \Huge{\color{Purple} 
\begin{align*}
h_t = g (W_{hh}h_{t-1}+W_{xh} x_t) + \textrm{b} & & \normalsize \begin{cases}
g = \textit{non-linear function} \\
W_{{\color{Cyan}hh}} = \textit{ takes an \textbf{h} and gives out an \textbf{h}} \\
W_{{\color{Cyan}xh}} = \textit{ takes an \textbf{x} and gives out an \textbf{h}}\\ 
W_{{\color{Cyan}xh}}, W_{{\color{Cyan}xh}} , \textbf{b}_n = \textit{ are constant with time}
\end{cases}
\end{align*}
}
$$

#### Description
> The **linear combination** of $\Huge{\color{Purple} h}$ and $\Huge{\color{Purple} x}$ and **weight matrix** $\Huge{\color{Purple} W}$ which we will multiply $\Huge{\color{Purple} h_{t-1}}$ of previous layer and some other **weight matrix** $\Huge{\color{Purple} W}$ which we will multiply $\Huge{\color{Purple} x_{t}}$ of same layer. 

### For _Output layer_ for any instance $\large{\color{Purple} \hat{y}_t}$  - 

$$ \Huge{\color{Purple} \hat{y_t} = g^* ( W_{yh} h_t + b)} $$

#### Description
> $\large{\color{Purple} g }$ needs not to be same as $\large{\color{Purple} g^* }$, even $\large{\color{Purple} g^* }$ not always be a **_Non-linear function_**. 

## Step #1: Weight calculation for hidden layers and output layers
To train an **RNN**, the trick is to **unroll it through time** and then simply use [_regular backpropagation_](https://github.com/iAmKankan/Neural-Network/blob/main/backpropagation/README.md). This strategy is called ***backpropagation through time (BPTT)***.

Like regular backpropagation, there is a **_first forward pass through the unrolled network_** (represented by the dashed arrows). 

**Note:** [Backpropagation is common in ANN or in Multi-Layer Perceptron](https://github.com/iAmKankan/Neural-Network/blob/main/backpropagation/README.md). 

Inorder to make the expressions simple we put _allias_ in the above two expressionas like 
> <img src="https://user-images.githubusercontent.com/12748752/188525079-36af334d-6d36-4550-8480-8094a409168a.png" width= 55%/>

### Now $\large{\color{Purple} h_t}$ and $\large{\color{Purple} \hat{y}_t}$  looks like- 
$$ {\color{Red} \boxed{\Huge{\color{Purple} \begin{align*}
& h_t = g (W h_{t-1}+ U x_t) + \textrm{b} \\
&  \hat{y_t} = g^* ( V h_t) \\
\end{align*}} 
}}
$$

#### Description
> The Matrixes $\large{\color{Purple} \textbf{W} }$, $\large{\color{Purple} \textbf{U} }$, $\large{\color{Purple} \textbf{V} }$ do not change with time (or across the layers); Meaning same values in each epoch.
> Where as in ANN those matrix changes its values. We need to update them. See for [Weight update in Backpropagation](https://github.com/iAmKankan/Neural-Network/blob/main/backpropagation/README.md#backpropagation-weight-update).

---
### ⚛️ How does RNN keep the context?

**Answer:** The following vectors <img src="https://latex.codecogs.com/svg.image?{\color{Purple}\textrm{W,&space;U,&space;V&space;}&space;}" title="https://latex.codecogs.com/svg.image?{\color{Purple}\textrm{W, U, V } }" align="center" /> do not change with time (or across the layers).

<p align="center">
<img src="https://user-images.githubusercontent.com/12748752/188549351-d5c3b022-9b5b-4b80-bf0c-ce57d3039940.png" width=50%/>
</p>

---

## Step #2: Loss calculation
For the Backpropagation we need to findout the derivative of the **_loss function_** let say $\large{\color{Purple} L }$ with each of the matrices $\large{\color{Purple} W }$, $\large{\color{Purple} U }$, $\large{\color{Purple} V}$ - 


$$
\Huge  {\color{Purple} \frac{\partial \textrm{L}}{\partial \textrm{W}},\ \frac{\partial \textrm{L}}{\partial \textrm{U}},\ \frac{\partial \textrm{L}}{\partial \textrm{V}}}  {\color{Purple} \Big \\{ \normalsize \textrm{For the Backprop we need to findout the gradient of 'L' with respect to each of the matrices} }
$$

### Recap of Loss function and $\large{\color{Purple}\partial L }$
* [The output sequence is evaluated using a cost function](https://github.com/iAmKankan/Neural-Network/tree/main/rnn#calculating-loss-in-rnn)

$$
\Huge{\color{Purple} \begin{align*}
\textbf{L} = \sum_{t=1}^{\textrm{T}} \textbf{L}_{t} & & \normalsize
\begin{cases} \textrm{where } T \textrm{ is the max time step} \\ 
\textrm{Summation of all the intermediate losses through the layers}\\
\end{cases}
\end{align*}}
$$

#### Description
> Summation of all the intermediate losses through the layers.

<p align="center">
<img src="https://user-images.githubusercontent.com/12748752/211141343-95154f22-bcaf-4a6a-a609-bd6d82e6b3e0.png" width=70%/>
</p>


---
#### Example #1: Lets consider local loss L3 and see how backprop works
$$
\Huge {\color{Purple} \frac{\partial \textrm{L}_3}{\partial \textrm{W}},\ \frac{\partial \textrm{L}_3}{\partial \textrm{U}},\ \frac{\partial \textrm{L}_3}{\partial \textrm{V}} }
$$

---

* We assume that _**g**_ is a **non-linear function** and-

$$
\Huge {\color{Purple} \hat{y_3} = g(V h_3) }
$$

* **Loss functions** used **least-squares error**

$$
\Huge {\color{Purple} \mathrm{L_3} = \frac{1}{2}(y_3 - \hat{y_3})^2 }
$$

![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

* We need to findout 

$$
\Huge {\color{Purple} 
\begin{align*}
& \frac{\partial \textrm{L}_3}{\partial \textrm{V}} & {\color{Black} \large \textrm{which can be expressed by- }} \\
& \frac{\partial \textrm{L}_3} {\partial \textrm{V}} = \frac{\partial \textrm{L}_3}{\partial \mathrm{\hat{y_3}}} \frac{\partial \mathrm{\hat{y_3}}}{\partial \textrm{V}} &\\
& \frac{\partial \textrm{L}_3} {\partial \textrm{V}} = - (\mathrm{y_3 - \hat{y_3}}) \mathrm{h_3} &\\
\end{align*}
}
$$

[**To be continued**]

![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

#### ⚛️ What is $\large \frac{\partial \textrm{L}_3}{\partial \textrm{W}}$ ?

$$
\Huge {\color{Purple} 
\begin{align*}
& \frac{\partial \textrm{L}_3} {\partial \textrm{W}} = \frac{\partial \textrm{L}_3}{\partial \mathrm{\hat{y_3}}} \frac{\partial \mathrm{\hat{y_3}}}{\partial \textrm{h}_3} \frac{\partial \textrm{h}_3}{\partial \textrm{W}}&\\
& \frac{\partial \textrm{L}_3} {\partial \textrm{V}} = - (\mathrm{y_3 - \hat{y_3}}) \mathrm{h_3} \mathrm{V}&\\
\end{align*}
}
$$

[**To be continued**]

## Step #3: 

<p align="center">
 <img src="https://user-images.githubusercontent.com/12748752/211140459-68704d15-7578-4136-9ede-77d418150d7c.png" width=50%/>
</p>

The gradients of that cost function are then **_propagated backward through the unrolled network_** (represented by the solid arrows). 

## <ins>Finally </ins>
* The **_model parameters are updated_** using the gradients computed during **BPTT**. 

**Note** that the gradients flow backward through all the outputs used by the cost function, not just through the final output (for example, in Figure the cost function is computed using the last three outputs of the network, <img src="https://latex.codecogs.com/svg.image?\textbf{Y}_{(2)},&space;\textbf{Y}_{(3)}\&space;and&space;\&space;\textbf{Y}_{(4)}" title="\textbf{Y}_{(2)}, \textbf{Y}_{(3)}\ and \ \textbf{Y}_{(4)}" />, so gradients flow through these three outputs, but not through **Y<sub>(0)</sub>** and **Y<sub>(1)</sub>** ). 
* Moreover, since the same parameters **W** and **b** are used at each time step, backpropagation will do the right thing and sum over all time steps.
* Fortunately, tf.keras takes care of all of this complexity for you



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


## Problems in Training Simple RNNs
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
1. Unstable Gradient

### Vanishing Gradients and TBPTT
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
#### [Vanishing Gradient and Exploding Gradient in Sigmoid Function ↗️](https://github.com/iAmKankan/Neural-Network/tree/main/activation_functions#vanishing-gradients-the-rnn-version-%EF%B8%8F)
#### ⚛️ Why we do BPTT or back-propagation-through-time
The basic issue for which we had to do **BPTT** was because **W, U, V** matrices were **constants across time**. Because of which you had sort of **recursive expressions** for the loss with respect to **W** and the loss with respect to **U**. 

The main issues that come up are 
* **gradient calculations** either **explode** or **vanish**, both of these are not ideal.
* The **gradient calculations are expensive**.

### The Solution
* The solution for **exploding gradients** is **_gradient clipping_**.
* The solution for **vanishing gradients** is alternate architectures **LSTM, GRU**.
* The solution for **expensive gradient calculations** is **_Truncated Back Propagation Through Time(TBPTT)_**.

#### Recap of BPTT 
<p align="center">
 <img src="https://user-images.githubusercontent.com/12748752/189465414-ed8ad124-93ed-4e56-a4ec-23f7c6d96c78.png" width=40%/>
 <br><ins><i><b>Typical RNN </b></i></ins>
 </p>

#### Total Loss- 

$$
\Huge{\color{Purple}
\begin{align*}
\mathrm{L}=\sum_{t=1}^{\mathrm{T}} \mathrm{L}_t
\end{align*}}
$$

#### ⚛️ When we are calculating _w_ if we are doing simple gradient descent 

$$
\Huge{\color{Purple}
\begin{align*}
& \mathrm{W}= \mathrm{W} - \alpha \frac{\partial\mathrm{L}}{\partial\mathrm{W}} \\
& \large \frac{\partial\mathrm{L}}{\partial\mathrm{W}} \Big \\}  \textrm{This term has to be calculated as: } \sum_{t=1}^{T} \frac{\partial \mathrm{L}_t}{\partial\mathrm{W}} \\
\end{align*}}
$$

We saw that you cannot simply calculate, let us say if I have **L<sub>3</sub>**, I cannot simply calculate<img src="https://latex.codecogs.com/svg.image?&space;\frac{\partial\mathrm{L}_3}{\partial\mathrm{W}}" title="https://latex.codecogs.com/svg.image? \frac{\partial\mathrm{L}_3}{\partial\mathrm{W}}" align="center"/> in the usual way besause 

$$\Huge{\color{Purple}
\begin{align*}
 &\frac{\partial\mathrm{L}_3}{\partial\mathrm{W}} \to \frac{\partial\mathrm{h}_3}{\partial\mathrm{W}}\to \frac{\partial\mathrm{h}_2}{\partial\mathrm{W}}\to \frac{\partial\mathrm{h}_1}{\partial\mathrm{W}}\\
& \large \textrm{They involve each other} \\
\end{align*}}
$$

Above is applicable for **U<sub>3</sub>**, as well 

$$\Huge{\color{Purple}
\begin{align*}
&\frac{\partial\mathrm{L}_3}{\partial\mathrm{U}} \to \frac{\partial\mathrm{h}_3}{\partial\mathrm{U}}\to \frac{\partial\mathrm{h}_2}{\partial\mathrm{U}}\to \frac{\partial\mathrm{h}_1}{\partial\mathrm{U}}\\
& \large \textrm{They involve each other} \\
\end{align*}}
$$

This is basically what we call **back propagation through time**, because **none of these terms is independent**. Now this kind of dependency creates several problems.

#### Explanation Why is BPTT is a problem:

<p align="center">
</a>
<br><ins><i><b>Heuristic Description.(rough)</b></i></ins>
</p>

$$\Huge{\color{Purple}
\begin{align*}
& h_t = tanh(W h_{t-1} + U x_t) \\
& \large \textrm{by cancelling } U x_t \textrm{ we get} \\
& h_t \sim  tanh(W h_{t-1}) [\large \textrm{eigenvalue need to understand}]\\
& h_t \sim  W h_{t-1} \large\textrm{ then}\\
& h_{t+1} \sim W^2 h_{t-1} \large [because \sim W^2 h_t \sim W^2 h_{t-1}] \\
& \large \textrm{In general- }\\
& \boxed{h_{t+1} \sim W^{n} h_{t} }\\
\end{align*}}
$$

> #### So, as you go through time, so the weight matrix keeps on constantly multiplying. So **_h<sub>3</sub>_** would be like **_W<sub>2</sub>h<sub>1</sub>_** and if I have something like **_h<sub>5</sub>_**, that would become **_W<sub>4</sub>h<sub>1</sub>_** so on and so forth.
 
Now, all these are heuristic arguments but it turns out to be a remarkably good approximations, unfortunately I cannot go further.
* But if I have norm(let us say 2 norm) of &parallel;**_h<sub>t+n</sub>_** &parallel; , notice **_h<sup>&#8407;</sup><sub>t</sub>_** is a vector.
* If I take its norm, it will be some factor times norm of **_h<sub>t</sub>_** (  &parallel;**_h<sub>t+n</sub>_** &parallel; &sim;  &parallel;**_h<sub>t</sub>_**&parallel; ) 
* Norm is a **scaler**, so this is a number, you are trying to find out the size of **_h<sub>t+n</sub>_**, that will be some number times **_h<sub>t</sub>_**
* And it turns out that it **scales** approximately as the **eigenvalues (&lambda;)** of **W<sup>n</sup>**. Like the following-

$$\Huge{\color{Purple}
\begin{align*}
& {\color{Cyan}\vec{{\color{Purple}h_{t+n}}}} \sim W^n {\color{Cyan}\vec{{\color{Purple}h_t}}} \\
& {\parallel \mathrm{h_{t+n}} \parallel }_2  \sim \lambda^n \parallel \mathrm{h_t}\parallel \\
\end{align*} }
$$

* Another way to see this is to assume that the **W** is diagonal, If **W** is diagonal, all it will have, **W<sup>n</sup>** will be, all its **eigenvalues** or all its diagonal terms to the power **n**. 
* Now which eigenvalue, we will see shortly. 
* The **eigenvalue** will either be the largest or the smallest.
  *  **The worst-case scenario** is if the **eigenvalue** will be the **_largest_**
  *  **The best case scenario** or the **smallest case scenario** is if the **eigenvalue** will be the **_smallest_**.

* Beacause of scaling as long as I use the same **W**, which I do for **RNN**, **throughout time**, what happens is these **vectors** constantly get **larger in magnitude** or constantly get **smaller in magnitude**.
* So, if you have a large number of time steps, this number, even if it is small, you know, for example even if it adds to **1.01**, over time it is going to get to be a huge number.(this is the power of the exponential function or of the power function)


$$\large{\color{Purple}
\begin{align*}
\textrm{For every 'h' }& & & \\
& \textrm{If } \huge{\mathrm{\lvert \lambda \rvert > 1}} & \textrm{ As 'n' increases }& \mathrm{\parallel h_{t+n}\parallel \to \infty }& \textrm{ (Become very large)}\\
& \textrm{If } \huge{\mathrm{\lvert \lambda \rvert < 1}} & \textrm{ As 'n' increases }& \mathrm{\parallel h_{t+n}\parallel \to 0 }\\
\end{align*} }
$$

* Now this is simply for **h**, you can show that and I would request you to try this out by looking at the expressions in **BPTT**, the similar arguments hold true for $\frac{\partial \mathrm{L_3}}{\partial \mathrm{W}}$ also.

$$\large{\color{Purple}
\begin{align*}
\boxed{\frac{\partial \mathrm{L_3}}{\partial \mathrm{W}} \to \frac{\partial \mathrm{h_3}}{\partial \mathrm{W} } } \to W \frac{\partial \mathrm{L_2}}{\partial \mathrm{W}}\\
\end{align*} }
$$


$$\Huge{\color{Purple}
\begin{align*}
& \parallel \frac{\partial \textrm{L}}{\partial \textrm{W}} \parallel \to \infty & \large \textbf{Exploding Gradient} \\
& \parallel \frac{\partial \textrm{L}}{\partial \textrm{W}} \parallel \to 0 & \large \textbf{Vanishing Gradient} \\
\end{align*}
\left \\} \begin{matrix}
  \\
 \large \textrm{Very Difficult to train}\\
  \\
\end{matrix}\right.
}
$$

### ⚛️ Why the big number is a problem _since &infin; means a big number_? 
**Answer:** These is a problems because obviously you are never going to get exactly **&infin;** because you are still dealing with finite number. But the problem is the moment it goes about the largest number that your machine can calculate, it will actually show you **NAN**, not a number or it will show you **&infin;**, so on and so forth. So really speaking, **finite preciation machines** cannot handle **exploding gradient**.

### ⚛️ Why the small number is a problem?
**Answer:** Similarly you will never actually go to **0**. If you do like **0:99<sup>1000</sup>** (because of **finite preciation machines**), it will be very very very small number. But the problem is it might actually becomes smaller than **10<sup>-16</sup>**, which is the smallest number that you can represent accurately. So at that point you will no longer train, so that will be called **saturation**. So you will get a very small gradient and that is practically gone. 

There is another problem, notice this **tanh**, even the **tanh** is being repeated multiple times. So you have **h<sub>t</sub>** **= tanh(W h<sub>t-1</sub>), h<sub>t+1</sub>** will be **tanh(h<sub>t</sub>)**, so you have **tanh<sup>2</sup>**, similarly you will have **tanh<sup>3</sup>**.

<p align="center">
 
 <img src="https://user-images.githubusercontent.com/12748752/189503599-5e3436e7-f19d-4f0e-957b-bad9d2cd1398.png" width=40%/>
 <br><ins><i><b>tanh<sup>2</sup> is flat, tanh<sup>3</sup> will look even flatter</b></i></ins>
 
</p>

Look at the  **tanh**, **tanh<sup>2</sup>**, **tanh<sup>2</sup>** will look even flatter. And if you take **tanh<sup>100</sup>**, it will look even smaller and notice in all these cases, gradients become **flatter** and **flatter** and **flatter** and they become very small.

Now, all these problems put together lead to these 2 issues. The **tanh**, repeated **tanh** problem will lead only to the **_vanishing gradient_** issue but large number of players can either lead to **_exploding gradient_** or it can lead to **_vanishing gradient_**, both of these make training very dificult.

### ⚛️ _Gradient clipping_ for exploding gradients
**Answer:**  It is very simple, we decide on a **maximum allowable gradient size**. What do I mean by value of gradient? **Gradient is a vector**, so you cannot give it a value, **you can however give a value to _norm_ of gradient**.
* Say-

$$\Huge{\color{Purple}
\begin{align*}
&  \frac{\partial \textrm{L}}{\partial \textrm{W}} = \vec{g} & max \parallel \vec{g} \parallel = G_{max} \\
\end{align*}
}
$$

#### Gradient Descent Calculate for $\large{\color{Purple}\vec{g}}$
$$\Huge{\color{Purple}
\begin{align*}
& \textit{If  }  \parallel \vec{g} \parallel < G_{max} \large \textrm{[ Proceed as usual]} \\
& \textit{If not, }  \parallel \vec{g^\*} \parallel =  \frac{\vec{g}}{\parallel \vec{g} \parallel} G_{max} \\
\end{align*}
}
$$

*  My new gradient $\vec{g^\*}$ is in the same direction as the gradient you calculated but I am cutting down its size

### ⚛️ Solution for Vanishing gradients 
**Answer:** Unfortunately no such simple solution exists. LSTM, GRU is the result

### ⚛️ Solution for **expensive gradient Computation** **_Truncated Back Propagation Through Time(TBPTT)_**
**Answer:** This solution kind of handles to a certain extent, both the **vanishing gradient** and the **exploding gradient** problems. 

So, if we have data with thousands and thousands of time steps. And you want to calculate back **propagation through time**. Now how would you do it? 
#### Step #1
Forward propagate through the whole thing, calculate the whole of <img src="https://latex.codecogs.com/svg.image?{\color{Purple}&space;\mathbf{L_T}}" title="https://latex.codecogs.com/svg.image?{\color{Purple} \mathbf{L_T}}" align="center"/> 
#### Step #2
Then you will **back propagate** through the whole thing. 

Now we have **65,000 time sequences**. If you go back for the full thing and come back through the full thing, by that time almost any correction you give will lead to vanishing or exploding gradient problems, plus it would become potentially very **expensive** just to do one gradient update.

So the solution to that is **truncated back propagation through time**.

Since throughout the RNN network you are going to get exactly the same **_W_**. So, instead of training for the whole **sequence**, you split it up into many mini batches( similar like mini batch gradient descent).

I will **forward propagate** through **first 2 steps** and **back propagate** through **2 steps**, this is one possibility. Since **_W_** is the same everywhere. So I will get some new updated **_W_**. 

Next I **forward propagates** through another 2 steps, back propagates through 2 steps, my W is now updated , okay. Now when the W is updated, I will forward propagate through the whole thing, okay. So I keep on doing this.

If you back propagates through a small amount of data, your gradient will neither blow up, nor will it **vanish**. Now what is a good rule of thumb? It is actually hard to say for some problems, hundred steps are good for some problems, 10, 20 steps are good, etc.

### Deep RNNs
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

<p align="center">
  
  <img src="https://user-images.githubusercontent.com/12748752/189594861-791518a2-c38d-4583-862d-b4f2939750c7.png" width=70%/>
  <br><ins><i><b> Deep RNN</b></i></ins>
  
</p>

The **deep RNNs** are particularly important in **language processing** especially in **language translation**.

$${\color{Purple}
\begin{align*}
& \huge h_t^{(l)} = tanh \Big\( W^{(l)}h_{t-1}^{(l)}+U^{(l)}h_{t}^{(l-1)} \Big \)\\
\end{align*}
}
$$


Now, what are deep RNNs let us look at just one of these if I look at one of these within the **RNN** it is just an ANN as we saw with normal RNNs. In a **normal RNN** all you had was **one input layer**, **one hidden layer** and **one output layer**. In a deep RNN all you do is that one single layer of the RNN actually become a deep neural network that is the only difference between a deep RNN and a normal RNN.

![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
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

![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)








![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

## Bibliography
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition by Aurélien Géron**
* NPTEL(Dr. Balaji Srinivasan)
* [YouTube 1](https://www.youtube.com/watch?v=0XdPIqi0qpg)
