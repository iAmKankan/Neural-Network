## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

## Long Short Term Memory 
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
_**The challenge to address long-term information preservation and short-term input skipping in latent variable models has existed for a long time. One of the earliest approaches to address this was the long short-term memory (LSTM)**_
* Usually just called "LSTMs" – are a special kind of RNN, capable of learning long-term dependencies. 
* LSTMs are explicitly designed to avoid the long-term dependency problem. 
* Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!

## Gated Memory Cell
Arguably LSTM’s design is inspired by logic gates of a computer. LSTM introduces a memory cell (or cell for short) that has the same shape as the hidden state (some literatures consider the memory cell as a special type of the hidden state), engineered to record additional information. To control the memory cell we need a number of gates. One gate is needed to read out the entries from the cell. We will refer to this as the output gate. A second gate is needed to decide when to read data into the cell. We refer to this as the input gate. Last, we need a mechanism to reset the content of the cell, governed by a forget gate. The motivation for such a design is the same as that of GRUs, namely to be able to decide when to remember and when to ignore inputs in the hidden state via a dedicated mechanism. Let us see how this works in practice.

## Why LSTM needed?
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* Due to the transformations that the data goes through when traversing an RNN, some information is lost at each time step. 
* After a while, the RNN’s state contains virtually no trace of the first inputs. 
* To tackle this problem, the LSTM is comming into the picture.

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




## References
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
