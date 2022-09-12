## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

## Why Long Short Term Memory (LSTM) needed?
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
In **GRU**, by **introducing new weights** and a **slightly more complicated structure** we could probably handle the **vanishing gradient issue** of **_Vanilla RNN_**. **GRU** or the **simplified GRU** we had something like **h<sub><i>t</i></sub>** = _**f**_ **&odot;** **h<sub><i>t -</i>1</sub>** **+** (**1-** _**f**_) **&odot;** **_g_**, where **g** was the output of the **vanilla RNN** and the idea there was to **retain some portion of your old calculations** into other **new ones**. In the case of **LSTM**, It is using a separate memory cell **C<sub><i>t</i></sub>** all together.

### The formulation of LSTM
$$\Huge{\color{Purple}
\begin{align*}
\boxed{\begin{align*}
& \mathbf{C_{\textit{t}}} = f \odot \mathbf{C_{\textit{t-1}}} + i \odot g  \\
& \mathbf{h_{\textit{t}}} = \mathbf{O} \odot tanh (\mathbf{C_{\textit{t}}})  \\
& \normalsize\mathrm{C_{\textit{t}}:}\textit{ memory cell of LSTM}
\end{align*}} 
\large\begin{cases}
& \textrm{O} = \sigma(z_{\textrm{O}}) & \textrm{ output gate}\\
& \textrm{f} = \sigma(z_{\textrm{f}}) & \textrm{ forget gate}\\
& \textrm{i} = \sigma(z_{\textrm{i}}) & \textrm{ input gate}\\
& \textrm{g} = tanh(z_{\textrm{g}}) & \textrm{ Vanilla RNN output}\\
\end{cases}
\large\begin{cases}
& z_\textrm{O} = W_{\textrm{O}}h_{\textrm{t-1}} + U_{\textrm{O}}x_{\textrm{t-1}}\\
& z_\textrm{f} = W_{\textrm{f}}h_{\textrm{t-1}} + U_{\textrm{f}}x_{\textrm{t-1}}\\
& z_\textrm{i} = W_{\textrm{i}}h_{\textrm{t-1}} + U_{\textrm{i}}x_{\textrm{t-1}}\\
& z_\textrm{g} = W_{\textrm{g}}h_{\textrm{t-1}} + U_{\textrm{g}}x_{\textrm{t-1}}\\
\end{cases}
\end{align*}
}
$$

Now if you see LSTM, it has how many unknowns you know what ever be the size of the W matrices you have **8** unknown weight matrices, just for comparison plain vanilla RNN just has **two** weight matrices GRU has **4** and **6**.

**LSTM** typically can **train or retain** **non vanishing gradient** for **greater number of layers** compared to **GRU** and GRU typically can retain **greater number of layers** compared to **vanilla RNN**

$$\Huge{\color{Purple}
\begin{align*}
& \mathbf{LSTM}  \mathbf{\Big\>\Big\>} \mathbf{GRU} \mathbf{\Big\>\Big\>} \mathbf{Vanilla RNN} &\\
\end{align*}
\large\begin{cases}
& \textrm{Better training performence in non-vanishing layers}\\
& \textrm{Better in complitional time}\\
\end{cases}
}
$$

The depth of the architecture can be greater with LSTM compared to GRU and that compared to vanilla RNN and that you have to balance it against typically the number of weights that you have to train, so of course this is also true of non-vanishing layers that you can train plus time taken for computation.

<p align="center">
  
  <img src="https://user-images.githubusercontent.com/12748752/189539087-6aad2553-7d7d-4507-ba98-43336ed849e0.png" width=70% />
  <br><ins><i><b> Schematic Long Short Term Memory (LSTM) </b></i></ins> 
  
</p>

#### Stage #1
* Now we have not only **h<sub><i>t-1</i></sub>** and **x<sub><i>t</i></sub>** coming into the box, which is finally going to spit out **h<sub><i>t</i></sub>**, 
* Next we have memory cell or memory computation, **C<sub><i>t-1</i></sub>** coming in, and **C<sub><i>t</i></sub>** going out.
* So, **C<sub><i>t</i></sub>** progresses **h<sub><i>t</i></sub>** progresses, and there is some processing that happens inside which was given by our formulation above. 
#### Stage #2
* Next, **h<sub><i>t-1</i></sub>** and **x<sub><i>t</i></sub>** combine to give our **vanilla RNN** output **_g_**. 
* Now, **C<sub><i>t-1</i></sub>**, there is a **valve** (**&in; [0, 1]**) here. 
* So, the **input gate** **_i_** gets multiplied by **_g_**.
* And the **forget gate** **_f_** gets multiplied by **C<sub><i>t</i></sub>**.
* Above two combine and this is what gives us **C<sub><i>t</i></sub>** as **output**.
* At the same time the same **C<sub><i>t</i></sub>** comes down and run it through a **_tanh_**. 
* then again it run through the **output gate** **_O_** and what you get is **h<sub><i>t</i></sub>**.

### Why LSTM Works
#### In case of _Vanishing Gradient_
#### GRU-

$$\Huge{\color{Purple}
\begin{align*}
& h_t = f \odot h_{t-1}  + (1 - f) \odot g  & \Big \\{ \large g = tanh (z_g), &  f = \sigma (z_f),\\
\end{align*}
}
$$

#### Why was it that the gradient was vanishing in there first place?
**Answer:** You can think of this **f &odot; h<sub>t-1</sub>**, **_f_** as _weight matrix_ **W** and if this weight matrix is **multiplies itself multiple times through multiple layers** there [**eigenvalue**](https://github.com/iAmKankan/Mathematics/blob/main/LinearAlgebra/matrix.md#eigen-decomposition-or-matrix-factorization) when it raise to the power **n** and if it is **less than one** it can actually go to **zero**, **that was the basic problem**. When this goes to **W<sup>n</sup>** it went like **&lambda; <sup>n</sup>**,

#### Why it helps GRU
**Answer:** 
* When **_f_** turns as _weight matrix_ **W** then **(1- f)** also turns to **I-W**(Identity matrix minus **W**)
* So is W goes to small then **I-W** becomes correspondingly large.
* So the 1st term and the 2nd term balance out
* **Alternate path for the gradient**- This plus **(+)** is what makes things work, why it is plus makes thing work because just like ResNet and Alexnet, there is an alternate path way for the gradient that is when you are doing back prop it can either go directly through this or it can go through this.

<p align="center">
  
  <img src="https://user-images.githubusercontent.com/12748752/189553007-3e45b556-d0f5-4633-a31b-39d0108a008d.png" width=50%/>
    <br><ins><i><b> Alternate Ways in AlexNet and LSTM </b></i></ins> 
  
</p>

Whenever you have training problems try and provide alternate pathway try and provide some skips connections try and provide some difierent way to actually train, and that is really what as we understand it what happens even within simplified GRU or within LSTM.


## Long Short Term Memory 
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
_**The challenge to address long-term information preservation and short-term input skipping in latent variable models has existed for a long time. One of the earliest approaches to address this was the long short-term memory (LSTM)**_
* Usually just called "LSTMs" – are a special kind of RNN, capable of learning long-term dependencies. 
* LSTMs are explicitly designed to avoid the long-term dependency problem. 
* Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!

## Gated Memory Cell
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* Arguably LSTM’s design is inspired by logic gates of a computer. LSTM introduces a memory cell (or cell for short) that has the same shape as the hidden state (some literatures consider the memory cell as a special type of the hidden state), engineered to record additional information. 
* To control the memory cell we need a number of gates. 
   * One gate is needed to read out the entries from the cell. We will refer to this as the **_output gate_**. 
   * A second gate is needed to decide when to read data into the cell. We refer to this as the input gate. Last, we need a mechanism to reset the content of the cell, governed by a forget gate. The motivation for such a design is the same as that of GRUs, namely to be able to decide when to remember and when to ignore inputs in the hidden state via a dedicated mechanism. Let us see how this works in practice.

## Input Gate, Forget Gate, and Output Gate
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* Just like in GRUs, the data feeding into the LSTM gates are the input at the current time step and the hidden state of the previous time step, as illustrated in Fig. 9.2.1. They are processed by three fully-connected layers with a sigmoid activation function to compute the values of the input, forget. and output gates. As a result, values of the three gates are in the range of  (0,1) .

> ### A type of RNN
* All recurrent neural networks have the form of a chain of repeating modules of neural network. 
* In standard RNNs, this repeating module will have a very simple structure, such as a single **_tanh_** layer.
<img src="https://user-images.githubusercontent.com/12748752/154867618-8e3864f4-8885-454e-9b20-13f8de785342.png" width=50% />

**The repeating module in a standard RNN contains a single layer.**

> ### LSTM
* LSTMs also have this chain like structure, but the repeating module has a different structure. 
* Instead of having a single neural network layer, there are four, interacting in a very special way.
<img src="https://user-images.githubusercontent.com/12748752/155165190-42982db6-82f5-4dd0-812e-ada067cea977.png" width=50% />

**The repeating module in an LSTM contains four interacting layers.**

### The Core Idea Behind LSTMs
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
---
## How does an LSTM cell work?
<img src="https://user-images.githubusercontent.com/12748752/165821166-85670ec0-8c2a-409d-a4dc-e878366adc24.png" width=90% />

The LSTM cell is split into two vectors: 
  * ***h<sub>(t)</sub>*** (the short-term state)
  * ***c<sub>(t)</sub>*** (the long-term stat).

The key idea is that the network can learn _what to store in the long-term state_, _what to throw away_, and _what to read from it_. 
#### Section 1: 
* As the long-term state **_c<sub>(t-1)</sub>_** traverses the network from left to right, it first goes through a _forget gate_ (**&otimes;**),  dropping some memories, and then it adds some new memories via the _addition operation_ **&oplus;** (which adds the memories that were selected by an input gate). 
* The result **c<sub>(t-1)</sub>** is sent straight out, without any further transformation. 
* So, at each time step, some memories are dropped and some memories are added. After the addition operation, the long-term state is copied and passed through the **tanh** function, and then the result is filtered by the output gate. This produces the short-term state **h<sub>(t)</sub>** (which is equal to the cell’s output for this time step, **y<sub>(t)</sub>** ). 

#### Section 2:
The current _input vector_ **x<sub>(t)</sub>** and the previous _short-term state_ **h<sub>(t-1)</sub>** are fed to four different fully connected layers. They all serve a different purpose:
   * The main layer is the one that outputs **g<sub>(t)</sub>** . It has the usual role of analyzing the current inputs **x<sub>(t)</sub>** and the previous (short-term) state **h<sub>(t-1)</sub>**. 
   * In a basic cell, there is nothing other than this layer, and its output goes straight out to **y<sub>(t)</sub>** and **h<sub>(t)</sub>** . _In contrast, in an LSTM cell this layer’s output does not go straight out, but instead its most important parts are stored in the long-term state (and the rest is dropped)._
   * The three other layers are **_gate controllers_**. Since they use the _logistic activation function_, their outputs range from **0** to **1**. As you can see, their outputs are fed to element-wise multiplication operations, so if they output **0s** they close the gate, and if they output **1s** they open it. Specifically:
     * **The forget gate** (controlled by **f<sub>(t)</sub>** ) controls which parts of the long-term state should be erased. 
     * **The input gate** (controlled by **i<sub>(t)</sub>** ) controls which parts of **g<sub>(t)</sub>** should be added to the long-term state. 
     * **The output gate** (controlled by **o<sub>(t)</sub>** ) controls which parts of the long-term state should be read and output at this time step, both to **h<sub>(t)</sub>** and to **y<sub>(t)</sub>** . 
     
> #### In short, an LSTM cell can learn to recognize 
* an important input (that’s the role of the _input gate_), 
* store it in the _long-term state_, 
* preserve it for as long as it is needed (that’s the role of the _forget gate_), and extract it whenever it is needed. 

This explains why these cells have been amazingly successful at capturing long-term patterns in `time series`, `long texts`, `audio recordings` and more. 

<img src="https://latex.codecogs.com/svg.image?\\\mathbf{g_{(t)}&space;=&space;tanh(W_{xg}^\top&space;x_{(t)}&plus;W_{hg}^\top&space;x_{(t-1)}&plus;b_g&space;)}\\&space;\\\mathbf{i_{(t)}&space;=&space;\sigma(W_{xi}^\top&space;x_{(t)}&plus;W_{hi}^\top&space;x_{(t-1)}&plus;b_i&space;)}\\&space;\\\mathbf{f_{(t)}&space;=&space;\sigma(W_{xf}^\top&space;x_{(t)}&plus;W_{hf}^\top&space;x_{(t-1)}&plus;b_f&space;)}&space;\\&space;\\\mathbf{o_{(t)}&space;=&space;\sigma(W_{xo}^\top&space;x_{(t)}&plus;W_{ho}^\top&space;x_{(t-1)}&plus;b_o&space;)}&space;\\&space;\\\mathbf{c_{(t)}&space;=&space;f_{(t)}&space;\otimes&space;c_{(t-1)}&space;&plus;&space;i_{(t)}&space;\otimes&space;g_{(t)}}\\&space;\\\mathbf{y_{(t)}&space;=&space;h_{(t)}&space;=&space;o_{(t)}&space;\otimes&space;tanh&space;(c_{(t)})}" title="https://latex.codecogs.com/svg.image?\\\mathbf{g_{(t)} = tanh(W_{xg}^\top x_{(t)}+W_{hg}^\top x_{(t-1)}+b_g )}\\ \\\mathbf{i_{(t)} = \sigma(W_{xi}^\top x_{(t)}+W_{hi}^\top x_{(t-1)}+b_i )}\\ \\\mathbf{f_{(t)} = \sigma(W_{xf}^\top x_{(t)}+W_{hf}^\top x_{(t-1)}+b_f )} \\ \\\mathbf{o_{(t)} = \sigma(W_{xo}^\top x_{(t)}+W_{ho}^\top x_{(t-1)}+b_o )} \\ \\\mathbf{c_{(t)} = f_{(t)} \otimes c_{(t-1)} + i_{(t)} \otimes g_{(t)}}\\ \\\mathbf{y_{(t)} = h_{(t)} = o_{(t)} \otimes tanh (c_{(t)})}" />


> #### **W<sub>xi</sub>** , **W<sub>xf</sub>** , **W<sub>xo</sub>** , **W<sub>xg</sub>** are the weight matrices of each of the four layers for their connection to the input vector **x<sub>(t)</sub>** .
> #### **W<sub>hi</sub>** , **W<sub>hf</sub>** , **W<sub>ho</sub>** , **W<sub>hg</sub>** are the weight matrices of each of the four layers for their connection to the previous short-term state **h<sub>(t-1)</sub>** .
> #### **b<sub>i</sub>** , **b<sub>f</sub>**, **b<sub>o</sub>** , **b<sub>g</sub>** are the bias terms for each of the four layers. Note that TensorFlow initializes **b<sub>f</sub>** to a vector full of **1s** instead of **0s**. This prevents forgetting everything at the beginning of training.


## References
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition by Aurélien Géron**.
* NPTEL(Dr. Balaji Srinivasan)

