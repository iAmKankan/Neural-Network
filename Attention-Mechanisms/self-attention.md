## Index:
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

## ⬛ Self-Attention at a High Level
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
### "_Attention is All You Need_" this paper first showed us the concept of “**_self-attention_**”.
### <ins>How does it work</ins>
#### Input sentence : "<img src="https://latex.codecogs.com/svg.image?{\color{Purple}&space;\textbf{\textrm&space;{The&space;animal&space;didn't&space;cross&space;the&space;street&space;because&space;it&space;was&space;too&space;tired}}}" title="https://latex.codecogs.com/svg.image?{\color{Purple} \textbf{\textrm {The animal didn't cross the street because it was too tired}}}" align="center" /> "

> #### What does “**it**” in this sentence refer to?  Is it referring to the street or to the animal? It’s a simple question to a human, but not as simple to an algorithm. 

* When the model is processing the word “**it**”, **self-attention** allows it to associate “**it**” with “**animal**”.
* As the model processes each _word_ (each position in the input sequence), **self attention** allows it to look at _other positions_ in the input sequence for clues that can help lead to a better encoding for this word.
* In **RNNs**, think of how maintaining a hidden state allows an **RNN** to incorporate its representation of previous words/vectors it has processed with the current one it’s processing. 
* Self-attention is the method the Transformer uses to bake the “**understanding**” of other relevant words into the one we’re currently processing.
<img src="https://user-images.githubusercontent.com/12748752/171284998-28585e5b-fd1b-4303-8be1-61938921aa75.png" width= 50%/>

## ⬛ Self-Attention in Detail
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
Let’s first look at how to calculate self-attention using vectors, then proceed to look at how it’s actually implemented – using matrices.
### <ins><i>Self-attention using vectors</i></ins>:
####  <ins>The first step</ins>:     
* In calculating self-attention is to create **three vectors** from each of the encoder’s input vectors (_in this case, the embedding of each word_). So for each word, we create a **Query vector**, a **Key vector**, and a **Value vector**. 
* These vectors are created by **multiplying the embedding** by **three matrices** that we trained during the training process.

> Notice that these new vectors are smaller in dimension than the embedding vector. Their dimensionality is 64, while the embedding and encoder input/output vectors have dimensionality of 512. They don’t HAVE to be smaller, this is an architecture choice to make the computation of multiheaded attention (mostly) constant.

<img src="https://user-images.githubusercontent.com/12748752/171284991-437781f4-ab77-47f9-bf73-64296d190174.png" width= 50%/>

<p align="center"><i><ins><b>Multiplying x1 by the WQ weight matrix produces q1, the "query" vector associated with that word. We end up creating a "query", a "key", and a "value" projection of each word in the input sentence.</b></ins></i></p>

### What are the “query”, “key”, and “value” vectors?
> They’re abstractions that are useful for calculating and thinking about attention. Once you proceed with reading how attention is calculated below, you’ll know pretty much all you need to know about the role each of these vectors plays.

#### <ins>The second step</ins>:
* In calculating **self-attention** is to calculate a score. 
* Say we’re calculating the **self-attention** for the _first word_ in this example, “**Thinking**”.
   * _We need to score each word of the input sentence against this word._ 
   * The score determines how much focus to place on other parts of the input sentence as we encode a word at a certain position.

The score is calculated by taking the dot product of the **_query vector_** with the key vector of the respective word we’re scoring. So if we’re processing the **self-attention** for the word in _position_ **#1**, the first score would be the dot product of **q1** and **k1**. The second score would be the dot product of **q1** and **k2**.

<img src="https://user-images.githubusercontent.com/12748752/171481656-e8230d49-a591-4563-9437-3c875973db1d.png" width=60%/>

#### <ins>The third and fourth steps</ins>:
These steps are to divide the scores by **8** (_the square root of the dimension of the key vectors used in the paper – 64. This leads to having more stable gradients. There could be other possible values here, but this is the default_), then pass the result through a **softmax** operation. **Softmax** **_normalizes_** the scores so they’re all positive and add up to **1**.

<img src="https://user-images.githubusercontent.com/12748752/171481652-6564521c-42a2-4c24-a773-c2f30a4eebeb.png" width=60%/>

This **softmax** score determines how much each word will be expressed at this position. Clearly the word at this position will have the highest **softmax** score, but sometimes it’s useful to attend to another word that is relevant to the current word.

#### <ins>The fifth step</ins>:
* It is to _multiply_ each **value vector** by the **softmax score** (_in preparation to sum them up_). 
* The intuition here is to keep intact the values of the word(s) we want to focus on, and drown-out irrelevant words (_by multiplying them by tiny numbers like 0.001, for example_).

#### <ins>The sixth step</ins>:
* The sixth step is to _sum up_ the **_weighted value vectors_**. This produces the **output** of the **self-attention** layer at this position (_for the first word_).
* That concludes the self-attention calculation. 
* The resulting vector is one we can send along to the **feed-forward neural network**. 
* In the actual implementation, however, this calculation is done in matrix form for faster processing. 
* So let’s look at that now that we’ve seen the intuition of the calculation on the word level.
<img src="https://user-images.githubusercontent.com/12748752/171481649-61f756c0-5ad7-49a1-9b01-10aa30776769.png" width=60%/>

### <ins><i>Matrix Calculation of Self-Attention</i></ins>:
#### <ins>The first step</ins>:
The first step is to calculate the Query, Key, and Value matrices. We do that by packing our embeddings into a matrix **X**, and multiplying it by the weight matrices we’ve trained (**WQ**, **WK**, **WV**).

<img src="https://user-images.githubusercontent.com/12748752/171481646-a1d00cb5-1915-4b40-bce6-d53202402b0e.png" width=60%/>
<p align="center" ><ins><i><b>Every row in the X matrix corresponds to a word in the input sentence. We again see the difference in size of the embedding vector (512, or 4 boxes in the figure), and the q/k/v vectors (64, or 3 boxes in the figure)</b></i></ins></p>

#### <ins>Finally</ins>:
Finally, since we’re dealing with **matrices**, we can condense steps two through six in one formula to calculate the outputs of the self-attention layer.

<img src="https://user-images.githubusercontent.com/12748752/171481640-20367973-a9d1-4512-89b3-5039371a6bd5.png" width=60%/>
<p align="center" ><ins><i><b>The self-attention calculation in matrix form</b></i></ins></p>

<img src="https://latex.codecogs.com/svg.image?\large&space;{\color{Purple}&space;\mathbf{Attention(Q,K,V)=&space;softmax(\frac{QK^T}{\sqrt{d_k}})V&space;}&space;}" title="https://latex.codecogs.com/svg.image?\large {\color{Purple} \mathbf{Attention(Q,K,V)= softmax(\frac{QK^T}{\sqrt{d_k}})V } }" />

## Multi-Headed Attention
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
The paper further refined the **self-attention** layer by adding a mechanism called “**multi-headed**” attention. This improves the performance of the attention layer in **two ways**:
* It expands the model’s ability to focus on different positions. Yes, in the example above, **z1** contains a little bit of every other encoding, but it could be dominated by the the actual word itself. It would be useful if we’re translating a sentence like “_The animal didn’t cross the street because it was too tired_”, we would want to know which word “it” refers to.
* It gives the attention layer multiple “**representation subspaces**”. As we’ll see next, with multi-headed attention we have not only one, but multiple sets of **Query**/**Key**/**Value** weight matrices (the Transformer uses eight attention heads, so we end up with eight sets for each encoder/decoder). Each of these sets is randomly initialized. Then, after training, each set is used to project the input embeddings (or vectors from lower encoders/decoders) into a different representation subspace.

<img src="https://user-images.githubusercontent.com/12748752/171481637-fadbcd00-b01d-4d2d-9ecc-9b7e43e7e39f.png" width=60%/>

<p align="center" ><ins><i><b>With multi-headed attention, we maintain separate Q/K/V weight matrices for each head resulting in different Q/K/V matrices. As we did before, we multiply X by the WQ/WK/WV matrices to produce Q/K/V matrices.</b></i></ins></p>

If we do the same self-attention calculation we outlined above, just eight different times with different weight matrices, we end up with eight different Z matrices
<img src="https://user-images.githubusercontent.com/12748752/171481622-3238a862-c630-4711-b13c-39215b45a2cb.png" width=60%/>

This leaves us with a bit of a challenge. The feed-forward layer is not expecting eight matrices – it’s expecting a single matrix (a vector for each word). So we need a way to condense these eight down into a single matrix.

How do we do that? We concat the matrices then multiply them by an additional weights matrix WO.

<img src="https://user-images.githubusercontent.com/12748752/171481663-3a2c2ffd-8c20-4aab-a9e7-7758d8f11e38.png" width=60%/>

That’s pretty much all there is to multi-headed self-attention. It’s quite a handful of matrices, I realize. Let me try to put them all in one visual so we can look at them in one place

<img src="https://user-images.githubusercontent.com/12748752/171481659-b9a340f1-5cec-4479-aadc-fe73acec3283.png" width=60%/>

Now that we have touched upon attention heads, let’s revisit our example from before to see where the different attention heads are focusing as we encode the word “it” in our example sentence:

<img src="https://user-images.githubusercontent.com/12748752/171727414-6f6a4a3f-cb1d-4910-bb68-5f72c66b8c05.png" width=40%/>
<p align="center"><ins><b><i>As we encode the word "it", one attention head is focusing most on "the animal", while another is focusing on "tired" -- in a sense, the model's representation of the word "it" bakes in some of the representation of both "animal" and "tired".</i></b></ins></p>

If we add all the attention heads to the picture, however, things can be harder to interpret:

<img src="https://user-images.githubusercontent.com/12748752/171727404-c3c0f61a-3653-4bf5-9d82-bd6f257e5339.png" width=40%/>


## ⬛ Self-Attention
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









