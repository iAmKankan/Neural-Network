## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

## Why LSTM needed?
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* Due to the transformations that the data goes through when traversing an RNN, some information is lost at each time step. 
* After a while, the RNN’s state contains virtually no trace of the first inputs. 
* To tackle this problem, various types of cells with long-term memory have been introduced. They have proven so successful that the basic cells are not used much anymore.

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
* As the long-term state **_c<sub>(t-1)</sub>_** traverses the network from left to right, it first goes through a _forget gate_ (**&otimes;**),  dropping some memories, and then it adds some new memories via the addition operation **&oplus;** (which adds the memories that were selected by an input gate). 
* The result **c<sub>(t-1)</sub>** is sent straight out, without any further transformation. 
* So, at each time step, some memories are dropped and some memories are added. After the addition operation, the long-term state is copied and passed through the **tanh** function, and then the result is filtered by the output gate. This produces the short-term state **h<sub>(t)</sub>** (which is equal to the cell’s output for this time step, **y<sub>(t)</sub>** ). 

#### Section 2:
The current _input vector_ **x<sub>(t)</sub>** and the previous _short-term state_ **h<sub>(t-1)</sub>** are fed to four different fully connected layers. They all serve a different purpose:
   * The main layer is the one that outputs **g<sub>(t)</sub>** . It has the usual role of analyzing the current inputs **x<sub>(t)</sub>** and the previous (short-term) state **h<sub>(t-1)</sub>**. 
   * In a basic cell, there is nothing other than this layer, and its output goes straight out to **y<sub>(t)</sub>** and **h<sub>(t)</sub>** . _In contrast, in an LSTM cell this layer’s output does not go straight out, but instead its most important parts are stored in the long-term state (and the rest is dropped)._
   * The three other layers are gate controllers. Since they use the logistic activation function, their outputs range from **0** to **1**. As you can see, their outputs are fed to element-wise multiplication operations, so if they output **0s** they close the gate, and if they output **1s** they open it. Specifically:
     * **The forget gate** (controlled by **f<sub>(t)</sub>** ) controls which parts of the long-term state should be erased. 
     * **The input gate** (controlled by **i<sub>(t)</sub>** ) controls which parts of g should be added to the long-term state. 
     * **The output gate** (controlled by **o<sub>(t)</sub>** ) controls which parts of the long-term state should be read and output at this time step, both to **h<sub>(t)</sub>** and to **y<sub>(t)</sub>** . 
     
In short, an LSTM cell can learn to recognize 
* an important input (that’s the role of the _input gate_), 
* store it in the _long-term state_, 
* preserve it for as long as it is needed (that’s the role of the _forget gate_), and extract it whenever it is needed. 

This explains why these cells have been amazingly successful at capturing long-term patterns in `time series`, `long texts`, `audio recordings` and more. 


## References
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
