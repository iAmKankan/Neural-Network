## Index:
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)


## ⬛ Self-Attention and Positional Encoding
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
In order to encode a sequence in deep learning, we often use **CNNs** or **RNNs**.

**Now**, with _attention mechanisms_, imagine that we feed **_a sequence of tokens_** into **_attention pooling_** so that the same set of tokens act as **queries**, **keys**, and **values**. Specifically, each **query** attends to all the **_key-value pairs_** and generates one attention output. 

Since the queries, keys, and values come from the same place, this performs **_self-attention_**, which is also called **_intra-attention_**.

We will discuss sequence encoding using self-attention, including using additional information for the sequence order.

### 🔲Self-Attention
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
Given a sequence of input tokens <img src="https://latex.codecogs.com/svg.image?\mathbf{x_1,...,x_n}" title="https://latex.codecogs.com/svg.image?\mathbf{x_1,...,x_n}" align="center"/> where any  <img src="https://latex.codecogs.com/svg.image?\mathbf{x_i&space;\in&space;\mathbb{R}^d&space;(1\leq&space;i\leq&space;n)}" title="https://latex.codecogs.com/svg.image?\mathbf{x_i \in \mathbb{R}^d (1\leq i\leq n)}" align="center"/>, its self-attention outputs a sequence of the same length <img src="https://latex.codecogs.com/svg.image?\mathbf{y_1,...,y_n}" title="https://latex.codecogs.com/svg.image?\mathbf{y_1,...,y_n}" /> where

<img src="https://latex.codecogs.com/svg.image?\large&space;{\color{Purple}&space;\mathbf{y_i\&space;=&space;\mathit{f}(x_i,(x_1,x_1),...,(x_n,x_n)&space;)\in&space;\mathbb{R}^d}" title="https://latex.codecogs.com/svg.image?\large {\color{Purple} \mathbf{y_i\ = \mathit{f}(x_i,(x_1,x_1),...,(x_n,x_n) )\in \mathbb{R}^d}" />

according to the definition of attention pooling <img src="https://latex.codecogs.com/svg.image?\mathit{f}" title="https://latex.codecogs.com/svg.image?\mathit{f}" align="center" /> in (10.2.4). Using multi-head attention, the following code snippet computes the self-attention of a tensor with shape (batch size, number of time steps or sequence length in tokens, ). The output tensor has the same shape.

### 🔲 Comparing CNNs, RNNs, and Self-Attention
Let us compare architectures for mapping a sequence of  tokens to another sequence of equal length, where each input or output token is represented by a -dimensional vector. Specifically, we will consider CNNs, RNNs, and self-attention. We will compare their computational complexity, sequential operations, and maximum path lengths. Note that sequential operations prevent parallel computation, while a shorter path between any combination of sequence positions makes it easier to learn long-range dependencies within the sequence [Hochreiter et al., 2001].
