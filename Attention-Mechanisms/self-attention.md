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


### <ins>_The Multi-Head Attention Layer — Self-Attention_</ins>
An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

<img src="https://user-images.githubusercontent.com/12748752/171082932-d39bfe4b-8ae6-4f93-b6ee-3b8d53e1a4bb.png" width=50%/>

<p align="center"><ins><i><b>(left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.</b></i></ins>.</p>



## Multi-Head Attention
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

* In practice, given the same set of **queries**, **keys**, and **values** we may want our model to combine knowledge from different behaviors of the same attention mechanism, such as capturing dependencies of various ranges (e.g., **shorter-range** vs. **longer-range**) within a sequence. 
* Thus, it may be beneficial to allow our attention mechanism _to jointly use different representation subspaces_ of **queries, keys, and values**.

To this end, instead of performing a single attention **pooling**, **queries**, **keys** and **values** can be transformed with  independently learned linear projections. Then these **_h_** projected queries, keys, and values are fed into **attention pooling in parallel**. 

In the end, **_h_** attention pooling outputs are concatenated and transformed with another learned linear projection to produce the final output. 

This design is called **_multi-head attention_**, where each **_h_** of the  attention pooling outputs is a head. Using **fully-connected layers** to perform **learnable linear transformations**.

<img src="https://user-images.githubusercontent.com/12748752/170055315-b69b2b13-f3a5-44c6-8a6a-6a4655359f80.png" width=60%/>
<p align="center"><ins><i><b>Multi-head attention, where multiple heads are concatenated then linearly transformed.</b></i></ins></p>

### Model
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
Let us formalize this model mathematically. 

Given a **query** <img src="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{q&space;\in&space;\mathbb{R}^{d_q}}&space;}" title="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{q \in \mathbb{R}^{d_q}} }" align="center"/> , **a key** <img src="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{k&space;\in&space;\mathbb{R}^{d_k}}&space;}" title="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{k \in \mathbb{R}^{d_k}} }" /> , and **a value**  <img src="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{v&space;\in&space;\mathbb{R}^{d_v}}&space;}" title="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{v \in \mathbb{R}^{d_v}} }" align="center" /> , each attention head <img src="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{h_i(i=1,\dots,&space;h)}&space;}" title="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{h_i(i=1,\dots, h)} }" align="center"/> is computed as


<img src="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{h_i=}\mathit{f}\mathbf{\left&space;(&space;W^{(q)}_{i}q,W^{(k)}_{i}k,W^{(v)}_{i}v&space;\right&space;)\in&space;\mathbb{R}^{p_v},}&space;}" title="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{h_i=}\mathit{f}\mathbf{\left ( W^{(q)}_{i}q,W^{(k)}_{i}k,W^{(v)}_{i}v \right )\in \mathbb{R}^{p_v},} }" />

where learnable parameters <img src="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf&space;{W^{(q)}_{i}&space;\in&space;\mathbb{R}^{p_q&space;\times&space;d_q},W^{(k)}_{i}&space;\in&space;\mathbb{R}^{p_k&space;\times&space;d_k}{\color{Black}\mathrm{\&space;and\&space;}&space;}W^{(v)}_{i}&space;\in&space;\mathbb{R}^{p_v&space;\times&space;d_v}&space;{\color{Black}\mathrm{\&space;and\&space;}&space;}\mathit{f}&space;}&space;}" title="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf {W^{(q)}_{i} \in \mathbb{R}^{p_q \times d_q},W^{(k)}_{i} \in \mathbb{R}^{p_k \times d_k}{\color{Black}\mathrm{\ and\ } }W^{(v)}_{i} \in \mathbb{R}^{p_v \times d_v} {\color{Black}\mathrm{\ and\ } }\mathit{f} } }" align="center"/> is attention pooling, such as additive attention and scaled dot-product attention in Section 10.3. The multi-head attention output is another linear transformation via learnable parameters <img src="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf&space;{W_{o}&space;\in&space;\mathbb{R}^{p_o&space;\times&space;hp_{v}}&space;}&space;}" title="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf {W_{o} \in \mathbb{R}^{p_o \times hp_{v}} } }" align="center"/> of the concatenation of <img src="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf&space;{h}}&space;" title="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf {h}} " align="center" /> heads:
 
<img src="https://latex.codecogs.com/svg.image?\large&space;{\color{Purple}\mathbf&space;{W_{o}}&space;&space;\begin{bmatrix}&space;\mathbf{h_1}\\&space;\vdots&space;\\&space;\mathbf{h_h}\end{bmatrix}\mathbf&space;{\in&space;\mathbb{R}^{p_o}&space;}&space;}&space;" title="https://latex.codecogs.com/svg.image?\large {\color{Purple}\mathbf {W_{o}} \begin{bmatrix} \mathbf{h_1}\\ \vdots \\ \mathbf{h_h}\end{bmatrix}\mathbf {\in \mathbb{R}^{p_o} } } " />
 
Based on this design, each head may attend to different parts of the input. More sophisticated functions than the simple weighted average can be expressed.
## Summary:
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* **_Multi-head attention_** combines knowledge of the same **attention pooling** via different representation subspaces of **queries**, **keys**, and **values**.
* To compute multiple heads of multi-head attention in parallel, proper tensor manipulation is needed. 









