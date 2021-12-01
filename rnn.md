## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* [Recurrent Neural Networks(RNN)](https://github.com/iAmKankan/Neural-Network/blob/main/rnn.md#recurrent-neural-networksrnn)
  * [Output of a recurrent layer for a single instance]()
  * [Memory Cells]()
  * [Different types of RNN based on Input and Output Sequences]()
## Recurrent Neural Networks(RNN)
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
> #### Feedforward Neural Networks
* Where the activations flow only in one direction, from the _`input layer`_ to the _`output layer`_ (a few exceptions are there). 

> #### Recurrent Neural Network(RNN)
* It looks very much like a feedforward neural network, except it also has connections pointing backward. 

<img src="https://user-images.githubusercontent.com/12748752/144027506-ea345023-d777-493c-88fd-fae058e70097.png" width=50%/>

> ####  <ins>A recurrent neuron (left)    |    unrolled through time (right)</ins>

<img src="https://user-images.githubusercontent.com/12748752/144035005-3e1f7cb9-3cd8-4f2f-9d11-98a4bfc61ce0.png" width=50%/>

> ####  <ins>A recurrent neuron (left)    |    unrolled through time (right)</ins>

* Let’s look at the simplest possible RNN, composed of one neuron receiving inputs, producing an output, and sending that output back to itself. 
* At each time step `t` (also called a frame), this recurrent neuron receives the inputs `x` as well as its own output from the previous time step, y .
* Since there is no previous output at the first time step, it is generally set to 0. 
* We can represent this tiny network against the time axis, as shown above.
* This is called _**`unrolling the network through time`**_ (it’s the same recurrent neuron represented once per time step).

> #### At each time step `t`, every neuron receives both the `input vector `<img src="https://latex.codecogs.com/svg.image?\textbf{x}_{(t)}" title="\textbf{x}_{(t)}" /> and the `output vector from the previous time step `<img src="https://latex.codecogs.com/svg.image?\textbf{y}_{(t-1)}" title="\textbf{y}_{(t-1)}" height=50%/>. 
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

### Different types of RNN based on Input and Output Sequences
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

<img src="https://user-images.githubusercontent.com/12748752/144064476-e96cb279-6ea7-4a07-aad8-d883b0549bc2.png" width=50% />

> #### <ins>Seq-to-seq (top left), seq-to-vector (top right), vector-to-seq (bottom left), and Encoder–Decoder (bottom right) networks </ins>

### Sequence-to-Sequence Network
* An RNN can simultaneously take a sequence of inputs and produce a sequence of outputs.
* This type of network is useful for predicting time series such as stock prices: you feed it the prices over the last _N_ days, and it must output the prices shifted by one day into the future (i.e., from _N – 1_ days ago to tomorrow).

### Sequence-to-Vector Network
* You could feed the network a sequence of inputs and ignore all outputs except for the last one. 
* For example, you could feed the network a sequence of words corresponding to a movie review, and the network would output a sentiment score (e.g., from –1 [hate] to +1 [love]).


###  Vector-to-Sequence Network
* Conversely, you could feed the network the same input vector over and over again at each time step and let it output a sequence.
*  For example, the input could be an image (or the output of a CNN), and the output could be a caption for that image.


### Encoder-Decoder Network
* Lastly, you could have a **sequence-to-vector network**, called an **`encoder`**, followed by a **vector-to-sequence network**, called a **`decoder`**. 
* For example, this could be used for translating a sentence from one language to another. 
  * You would feed the network a sentence in one language, the encoder would convert this sentence into a single vector representation, and then the decoder would decode this vector into a sentence in another language. 
  * This two-step model, called an `Encoder–Decoder`, works much better than trying to translate on the fly with a single `sequence-to-sequence RNN` (like the one represented at the top left): the last words of a sentence can affect the first words of the translation, so you need to wait until you have seen the whole sentence before translating it.

### Training RNNs
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* To train an RNN, the trick is to unroll it through time (like we just did) and then simply use regular backpropagation. 
* This strategy is called **`backpropagation through time (BPTT)`**.

## Bibliography
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition by Aurélien Géron**
