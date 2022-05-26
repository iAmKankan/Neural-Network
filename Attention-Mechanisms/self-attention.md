## Index:
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)


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
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
Let us compare architectures for mapping a sequence of  tokens to another sequence of equal length, where each input or output token is represented by a -dimensional vector. Specifically, we will consider CNNs, RNNs, and self-attention. We will compare their computational complexity, sequential operations, and maximum path lengths. Note that sequential operations prevent parallel computation, while a shorter path between any combination of sequence positions makes it easier to learn long-range dependencies within the sequence.

<img src="https://user-images.githubusercontent.com/12748752/170157769-9b87bc29-61f2-48b1-9419-16cabc397fb4.png" width=30% align="center"/> <img src="https://user-images.githubusercontent.com/12748752/170158752-5a05a0d6-39c3-4f64-b2e5-8b375c2296a7.png" width=55% align="top"/>

<ins><i><b>Comparing CNN (padding tokens are omitted), RNN, and self-attention architectures.</b></i></ins>

### 🔲 Positional Encoding
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
### How _Transformer approach_ differs from _Sequence Model approach_:
* Unlike _sequence models_, _transformer_ **does not take the input embeddings sequentially**; on the contrary, **it takes in all the embeddings at once.** 
* This allows for **parallelization** and **significantly decreases training time**. 
### Problems associated with this approach
* However, the drawback is that it **loses** the important information related to **_words' order_**. 
### To use the sequence order information
* We can inject **absolute** or **relative** _positional information_ by adding **positional encoding** to the input representations. 
* Positional encodings can be either learned or fixed. 
* For the model to preserve the advantage of _words' order_, **positional encodings** are added to the **input embeddings**. 
* Since the positional encodings and embeddings are summed up, they both have the same dimension of **_d = 512_**.
* There are different ways to choose positional encodings; the creators of the transformer used **sine** and **cosine** functions to obtain the positional encodings. 

> **_At even dimension_** _indices_ the **sine** formula is applied and **_at odd dimension_** _indices_ the **cosine** formula is applied. 

### A **fixed positional encoding** based on _`sine`_ and _`cosine`_ functions:
Suppose that the input representation  <img src="https://latex.codecogs.com/svg.image?\mathbf{X&space;\in&space;\mathbb{R}^{\mathit{n&space;\times&space;d}}}" title="https://latex.codecogs.com/svg.image?\mathbf{X \in \mathbb{R}^{\mathit{n \times d}}}" align="center"/> contains the **_d_**-dimensional embeddings for **_n_** tokens of a sequence. The positional encoding **X + P** outputs  using a positional embedding matrix <img src="https://latex.codecogs.com/svg.image?\mathbf{P&space;\in&space;\mathbb{R}^{\mathit{n&space;\times&space;d}}}" title="https://latex.codecogs.com/svg.image?\mathbf{P \in \mathbb{R}^{\mathit{n \times d}}}" align="center" /> of the same shape, whose element on the **_i<sup>th</sup>_** row and the **_(2j)<sup>th</sup>_** or the  column is **_(2j + 1)<sup>th</sup>_**

<img src="https://latex.codecogs.com/svg.image?\large&space;\\{\color{Blue}&space;\mathbf{p_{\mathit{i,2j}}&space;=&space;\sin&space;\left&space;(&space;\frac{\mathit{i}}{10000^{\mathit{2j/d}}}&space;\right&space;),&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;}}\\&space;\\{\color{Blue}&space;\mathbf{p_{\mathit{i,2j&plus;1}}&space;=&space;\cos&space;\left&space;(&space;\frac{\mathit{i}}{10000^{\mathit{2j/d}}}&space;\right&space;)&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;}}&space;" title="https://latex.codecogs.com/svg.image?\large \\{\color{Blue} \mathbf{p_{\mathit{i,2j}} = \sin \left ( \frac{\mathit{i}}{10000^{\mathit{2j/d}}} \right ), }}\\ \\{\color{Blue} \mathbf{p_{\mathit{i,2j+1}} = \cos \left ( \frac{\mathit{i}}{10000^{\mathit{2j/d}}} \right ) }} " />
 
<img src="https://latex.codecogs.com/svg.image?\large&space;&space;\mathbf{Or}" title="https://latex.codecogs.com/svg.image?\large \mathbf{Or}" align="justify"/>
 
 <img src="https://latex.codecogs.com/svg.image?\large&space;\\{\color{Purple}\mathbf{PE_{(pos,\&space;2i)}=sin\left&space;(pos/10000^{2i/d_{model}}\right&space;)}}&space;\\{\color{Purple}\mathbf{PE_{(pos,\&space;2i&plus;1)}=cos\left&space;(pos/10000^{2i/d_{model}}\right&space;)}}" title="https://latex.codecogs.com/svg.image?\large \\{\color{Purple}\mathbf{PE_{(pos,\ 2i)}=sin\left (pos/10000^{2i/d_{model}}\right )}} \\{\color{Purple}\mathbf{PE_{(pos,\ 2i+1)}=cos\left (pos/10000^{2i/d_{model}}\right )}}" />  <img src="https://latex.codecogs.com/svg.image?\begin{cases}{\color{Purple}\mathbf{pos}}=&space;\textrm{The&space;current&space;position}&space;\\&space;{\color{Purple}\mathbf{2i}}=&space;\textrm{Dimention&space;Index}&space;\\{\color{Purple}\mathbf{d_{model}}}=&space;\textrm{Dimention&space;=&space;512}&space;\end{cases}" title="https://latex.codecogs.com/svg.image?\begin{cases}{\color{Purple}\mathbf{pos}}= \textrm{The current position} \\ {\color{Purple}\mathbf{2i}}= \textrm{Dimention Index} \\{\color{Purple}\mathbf{d_{model}}}= \textrm{Dimention = 512} \end{cases}" align="right"/>

<ins> Positional Encodings Formula</ins>
 
 
 ### 🔲 Absolute Positional Information
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
To see how the monotonically decreased frequency along the encoding dimension relates to absolute positional information, let us print out the binary representations of **0,1,...,7**. As we can see, the lowest bit, the second-lowest bit, and the third-lowest bit alternate on every number, every two numbers, and every four numbers, respectively.
```Python
for i in range(8):
    print(f'{i} in binary is {i:>03b}')
```
#### Output
```
0 in binary is 000
1 in binary is 001
2 in binary is 010
3 in binary is 011
4 in binary is 100
5 in binary is 101
6 in binary is 110
7 in binary is 111
```
In binary representations, a higher bit has a lower frequency than a lower bit. Similarly, as demonstrated in the heat map below, the positional encoding decreases frequencies along the encoding dimension by using trigonometric functions. Since the outputs are float numbers, such continuous representations are more space-efficient than binary representations.
### 🔲  Relative Positional Information
Besides capturing absolute positional information, the above positional encoding also allows a model to easily learn to attend by relative positions. This is because for any fixed position offset **_&delta;_**, the positional encoding at position **_i + &delta;_** can be represented by a linear projection of that at position **_i_**.

This projection can be explained mathematically. Denoting **_&omega;<sub>j</sub>_** **= 1/10000** **_<sup>2j/d</sup>_**, any pair of ( **p<sub>(i,2j)</sub>,p<sub>(i,2j+1)</sub>** ) in [above](https://github.com/iAmKankan/Neural-Network/edit/main/Attention-Mechanisms/self-attention.md#a-fixed-positional-encoding-based-on-sine-and-cosine-functions) can be linearly projected to ( **p<sub>(i+&delta;,2j)</sub>,p<sub>(i+&delta;,2j+1)</sub>** )  for any fixed offset **_&delta;_**:


### 🔲 Conclision:
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* In self-attention, the queries, keys, and values all come from the same place.
* Both CNNs and self-attention enjoy parallel computation and self-attention has the shortest maximum path length. However, the quadratic computational complexity with respect to the sequence length makes self-attention prohibitively slow for very long sequences.
* To use the sequence order information, we can inject absolute or relative positional information by adding positional encoding to the input representations.
