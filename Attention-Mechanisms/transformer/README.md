## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)


## ⬛ A Deep Dive Into the Transformer
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
Unlike **self-attention** models that still rely on _RNNs_ for _input representations_, the **transformer model** is solely based on attention mechanisms without any _convolutional_ or _recurrent_ layer. 

Though originally proposed for **sequence to sequence learning** on **text data**, transformers have been pervasive in a wide range of modern deep learning applications, such as in areas of **language**, **vision**, **speech**, and **reinforcement learning**.

### The Model
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
<img src="https://user-images.githubusercontent.com/12748752/164050988-292430e3-b184-4942-a92e-f2297b1541d1.png" width=30%/>
<ins>The transformer architecture.</ins>

As we can see, the **transformer** is composed of an **_encoder_** and a **_decoder_**. Different from [Bahdanau attention](https://github.com/iAmKankan/Neural-Network/blob/main/Attention-Mechanisms/bahdanau_attention.md) for _sequence to sequence learning_, the **input** (_source_) and **output** (_target_) **sequence embeddings** are added with **positional encoding** before being fed into the **encoder** and the **decoder** that stack modules based on **self-attention**.

#### <ins>The architecture</ins>:
**On a high level**, 
The **_transformer encoder_** is a stack of _multiple identical layers_, where each layer has **two sublayers** (_either is denoted as sublayer_).
  * The first is a **multi-head self-attention pooling** and 
  * the second is a **positionwise feed-forward network**. 
* Specifically, in the encoder **self-attention**, **queries**, **keys** and **values** are all from the the _outputs_ of the previous encoder layer. 
* Inspired by the **ResNet** design, a **residual connection** is employed around **both sublayers**. 
* In the transformer, for any input <img src="https://latex.codecogs.com/gif.image?\dpi{110}{\color{Purple}&space;\mathbf{x&space;\in&space;\mathbb{R}^{d}}}&space;" title="https://latex.codecogs.com/gif.image?\dpi{110}{\color{Purple} \mathbf{x \in \mathbb{R}^{d}}} " align="center" /> at any position of the sequence, we require that <img src="https://latex.codecogs.com/gif.image?\dpi{110}{\color{Purple}&space;\mathbf{sublayer(x)&space;\in&space;\mathbb{R}^{d}}}&space;" title="https://latex.codecogs.com/gif.image?\dpi{110}{\color{Purple} \mathbf{sublayer(x) \in \mathbb{R}^{d}}} " align="center"/> so that the residual connection <img src="https://latex.codecogs.com/gif.image?\dpi{110}{\color{Purple}&space;\mathbf{x&plus;sublayer(x)&space;\in&space;\mathbb{R}^{d}}}&space;" title="https://latex.codecogs.com/gif.image?\dpi{110}{\color{Purple} \mathbf{x+sublayer(x) \in \mathbb{R}^{d}}} " align="center" /> is feasible. 
* This addition from the _residual connection_ is immediately followed by **_layer normalization_**. 
* As a result, the **transformer encoder** outputs a _d-dimensional vector_ representation for _each position_ of the input sequence.


The **_transformer decoder_** is also a stack of _multiple identical layers_ with **_residual connections_** and **_layer normalizations_**. 
* Besides the **two sublayers** described in the _encoder_, the **decoder** inserts **a third sublayer**, known as the **encoder-decoder attention**, between these two. 
* In the **encoder-decoder** attention, **queries** are from the _outputs_ of the _previous decoder layer_, and the **keys** and **values** are from the **transformer encoder outputs**. 
* In the decoder **self-attention**, **queries**, **keys**, and **values** are all from the the outputs of the **previous decoder layer**. 
* However, each position in the decoder is allowed to only attend to all positions in the decoder up to that position.
* This **masked attention** preserves the **auto-regressive** property, ensuring that the _prediction only depends on those output tokens that have been generated_.

We have already described and implemented multi-head attention based on [scaled dot-products](https://github.com/iAmKankan/Neural-Network/blob/main/Attention-Mechanisms/multi-head.md) and [positional encoding](https://github.com/iAmKankan/Neural-Network/blob/main/Attention-Mechanisms/self-attention.md#-positional-encoding). 

![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
Here, the classical example of translating from _English to French_ using the transformer is considered. Input sentence is as such <img src="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{I\&space;am\&space;a\&space;student}&space;}&space;&space;" title="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{I\ am\ a\ student} } " />, and the expected output is <img src="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{Je\&space;suis\&space;un\&space;\acute{e}tudiant}&space;}&space;" title="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{Je\ suis\ un\ \acute{e}tudiant} }" />.
### 🔲 The Encoder
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
We will start by taking a closer look at the encoder side, and discover what is happening at each step.

### _◼️ The input_
The raw data is an english text, however the transformer, like any other model, does not understand english language and, thus, the text is processed to convert every word into a **_unique numeric ID_**. 

This is done by using a specific dictionary of vocabulary, which can be generated from the training data, and that maps each word to a **numeric index**.
> Figure 2: Numerical Representation of the Raw Text (Image by Author)
 
### _◼️ Embedding Layer_
As in other models, the transformer uses learned embeddings to transform the input tokens into vectors of dimension **d = 512**. During training, the model updates the numbers in the vectors to better represent the input tokens.

> Figure 3: Embeddings of d=512 by The Embedding Layer (Image by Author)

### _◼️ Positional Encoding_
One aspect that differentiates the _transformer_ from previous _sequence models_ is that **it does not take the input embeddings sequentially**; on the contrary, **it takes in all the embeddings at once.** This allows for **parallelization** and **significantly decreases training time**. However, the drawback is that it loses the important information related to **words' order**. 

For the model to preserve the advantage of words' order, **positional encodings** are added to the **input embeddings**. Since the positional encodings and embeddings are summed up, they both have the same dimension of d = 512. There are different ways to choose positional encodings; the creators of the transformer used sine and cosine functions to obtain the positional encodings. 

_At even dimension_ indices the sine formula is applied and _at odd dimension_ indices the cosine formula is applied. 
> Figure 4, shows the formulas used to obtain the positional encodings.

<img src="https://latex.codecogs.com/svg.image?\large&space;\\{\color{Purple}\mathbf{PE_{(pos,\&space;2i)}=sin\left&space;(pos/10000^{2i/d_{model}}\right&space;)}}&space;\\{\color{Purple}\mathbf{PE_{(pos,\&space;2i&plus;1)}=cos\left&space;(pos/10000^{2i/d_{model}}\right&space;)}}" title="https://latex.codecogs.com/svg.image?\large \\{\color{Purple}\mathbf{PE_{(pos,\ 2i)}=sin\left (pos/10000^{2i/d_{model}}\right )}} \\{\color{Purple}\mathbf{PE_{(pos,\ 2i+1)}=cos\left (pos/10000^{2i/d_{model}}\right )}}" />  <img src="https://latex.codecogs.com/svg.image?\begin{cases}{\color{Purple}\mathbf{pos}}=&space;\textrm{The&space;current&space;position}&space;\\&space;{\color{Purple}\mathbf{2i}}=&space;\textrm{Dimention&space;Index}&space;\\{\color{Purple}\mathbf{d_{model}}}=&space;\textrm{Dimention&space;=&space;512}&space;\end{cases}" title="https://latex.codecogs.com/svg.image?\begin{cases}{\color{Purple}\mathbf{pos}}= \textrm{The current position} \\ {\color{Purple}\mathbf{2i}}= \textrm{Dimention Index} \\{\color{Purple}\mathbf{d_{model}}}= \textrm{Dimention = 512} \end{cases}" align="right"/>

<ins> Positional Encodings Formula</ins>
> Adding Positional Encodings to the Embeddings to Generate Positional Embeddings (ep) (Image by Author)


### 🔲 The Multi-Head Attention Layer — Self-Attention
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
> The Multi-Head Attention Layer (source)

There are two terms that need to be addressed in this section, _**self-attention**_ and **_multi-head_**.

### _◼️ Self-Attention:_
> #### The goal of **_self-attention_** is to <i><ins>capture contextual relationships between words</ins></i> in the sentence <i><ins>by creating an attention-based vector of every input word</ins></i>. 

The **_attention-based vectors_** help to understand how relevant every word in the input sentence is with respect to other words in the sentence (as well as itself).

The scale dot-product attention illustrated on the left side of figure 6 is applied to calculate attention-based vectors. Below is a detailed explanation of how these vectors are created from the positional embeddings.

The first step is to obtain the Query (Q), Keys (K) and Values (V). This is done by passing the same copy of the positional embeddings through three different linear layers, as seen in the figure below.

The second step is to create an attention filter from the Query (Q) and the Key (K). The attention filter will indicate how much each word is attended to at every position. It is created by applying the formula found in figure 8.


Figure 8: Generating an Attention Filter from the Query (Q) and the Key (K) (Image by Author)
Finally, to obtain an attention-based matrix (the final output of the self-attention layer), a matrix to matrix multiplication (matmul) is done between the attention filter and the Value (V) matrix generated previously. Resulting in the following final formula:

<img src="https://latex.codecogs.com/svg.image?\large&space;{\color{Purple}&space;\mathbf{Attention(Q,K,V)=&space;softmax(\frac{QK^T}{\sqrt{d_k}})V&space;}&space;}" title="https://latex.codecogs.com/svg.image?\large {\color{Purple} \mathbf{Attention(Q,K,V)= softmax(\frac{QK^T}{\sqrt{d_k}})V } }" />

### _◼️ Multi-Head Attention:_
As seen on the right side of figure 6, the scaled-dot product attention (i.e. self-attention) is not applied only once, but also several times (in the original paper it is applied 8 times). The objective is to generate several attention-based vectors for the same word. This helps the model to have different representations of the words' relations in a sentence.

The different attention-based matrices generated from the different heads are concatenated together and passed through a linear layer to shrink the size back to that of a single matrix.

#### Residual Connections, Add & Norm and the Feed-Forward Network

As one can notice from figure 1, the architecture includes residual connections (RC). The residual connections' goal is avoid loss of important information found in old information by allowing these information to bypass the multi-head attention layer. Therefore, the positional embeddings are added to the output of the multi-head attention and then normalized (Add & Norm) before passing it into a regular feed-forward network.

### 🔲 The Decoder
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
The decoder side has a lot of shared components with the encoder side. Therefore, this section will not be as detailed as the previous one. The main differences between the decoder and the encoder are that the decoder takes in two inputs, and applies multi-head attention twice with one of them being "masked". Also, the final linear layer in the decoder has the size (i.e. number of units) equal to the number of words in the target dictionary (in this case the french language dictionary). Each unit will be assigned a score; the softmax is applied to convert these scores into probabilities indicating the probability of each word to be present in the output.

### _◼️ The input_
The decoder takes in two inputs:

1. **The output of the encoder** — these are the keys (K) and the values (V) that the decoder performs multi-head attention on (the second multi-head attention in figure 1). In this multi-head attention layer, the query (Q) is the output of the masked multi-head attention.
2. **The output text shifted to the right** — This is to ensure that predictions at a specific position "i" can only depend at positions less than i (see figure 10). Therefore, the decoder will take in all words already predicted (position 0 to i-1) before the actual word to be predicted at position i. Note that the first generated word passed to the decoder is the token <start> and the prediction process continues until the decoder generates a special end token <eos>.


 <img src="https://user-images.githubusercontent.com/12748752/169290757-0d143632-7fd4-45af-857e-c25ee5db6ed9.gif"/>

 <ins>Outputs Shifted by Right as Inputs to the Decoder In the Inference Stage</ins>[...Image by 'Kheirie Elhariri'](https://towardsdatascience.com/attention-is-all-you-need-e498378552f9) 
 
 
 ### _◼️ Masked Multi-Head Attention_
The process of the masked multi-head attention is similar to that of the regular multi-head attention. The only difference is that after multiplying the matrices Q and K, and scaling them, a special mask is applied on the resulting matrix before applying the softmax (see left diagram of figure 6-Mask opt.). The objective is to have every word at a specific position "i" in the text to only attend to every other position in the text up until its current position included (position 0 until position i). This is important in the training phase, as when predicting the word at position i+1, the model will only pay attention to all the words before that position. Therefore, all positions after i, are masked and set to negative infinity before passing them to the softmax operation, which results in 0s in the attention filter (see figure 11).
 
 
 ### 🔲 The Conclusion
The Transformer model is a deep learning model that has been in the field for five years now, and that has lead to several top performing and state of the art models such as the BERT model. Giving its dominance in the field of NLP and its expanding usage in other fields such as computer vision, it is important to understand its architecture. This article covers the different components of the transformer and highlights their functionalities.
 
 

![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

## ⬛ Transformer
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
The architecture of the **_transformer_** model inspires from the **attention mechanism** used in the **encoder-decoder** architecture in **RNNs** to handle s**equence-to-sequence (seq2seq)** tasks.

But it eliminates the factor of **sequentiality**; meaning that, unlike _RNNs_, _the transformer does not process data in sequence (i.e. in order)_, which allows for **_more_** [**_parallelization_**](https://github.com/iAmKankan/MachineLearning_With_Python/blob/master/README.md#parallelization) and **_reduces training time_**.

### 🔲 Types of Transformers:
<img src="https://user-images.githubusercontent.com/12748752/167986704-ca5cb1fe-7730-4b61-a9f5-1aee7dbaa9e9.png" width=70%/>

### 🔲 As a _Blackbox_
In a machine translation application, it would take a sentence in one language, and output its translation in another.
<img src="https://user-images.githubusercontent.com/12748752/164888116-dfdb9a7f-60c1-4038-9bf6-3f47a133a244.png" width=80%/>

### 🔲 The _Encoder-Decoder_ Blocks
Inside the Transformer box _Encoder-Decoder_ Block and a connection between them.

<img src="https://user-images.githubusercontent.com/12748752/164888115-281a74f2-971d-4eb3-8bcb-0bb58b35727b.png" width=80% />

### 🔲 Inside  _Encoder-Decoder_ Stack
The encoding component is a stack of encoders (the paper stacks **six** of them on top of each other – there’s nothing magical about the number **6**, one can definitely experiment with other arrangements). The decoding component is a stack of decoders of the same number.
<img src="https://user-images.githubusercontent.com/12748752/167968727-488848ff-40d1-49a9-99ad-61287bebba3e.png" width=80%/>

### 🔲 The individual _Encoder-Decoder_ Blocks
<img src="https://user-images.githubusercontent.com/12748752/164050988-292430e3-b184-4942-a92e-f2297b1541d1.png" align="right" width=25% />

The encoders are all identical in structure (yet they do not share weights). Each one is broken down into two sub-layers:
#### Each Encoder block having two components
  1) A **Feedforward layer** or a **place holder**(**_RNN, LSTM, GRU_**)
  2) A **Self-attention** layer
The encoder’s inputs first flow through a **self-attention layer** – _a layer that helps the encoder look at other words in the input sentence as it encodes a specific word._

The outputs of the **self-attention layer** are fed to a **feed-forward** or a **place holder**(**_RNN, LSTM, GRU_**) neural network. The exact same neural network is independently applied to each position.
#### Each Decoder block having three components
  1) A **Feedforward layer** or a **place holder**(**_RNN, LSTM, GRU_**)
  2) A **Self-attention** layer
  3) A **Encoder-Decoder Attention** layer
  
The decoder has both those layers, but between them is an **attention layer** that helps the decoder focus on relevant parts of the input sentence (_similar what attention does in seq2seq models_).

<img src="https://user-images.githubusercontent.com/12748752/168034980-004fd235-28cb-4831-9523-76480b411e11.png" width=80%/> 

<img src="https://user-images.githubusercontent.com/12748752/168195356-8a08298c-9157-4656-9464-0dd4f7d56145.png"/>


### 🔲 Embedding Algorithm
Let’s start to look at the various **vectors**/**tensors** and how they flow between the _above components_ to turn the _input of a trained model into an output_.

As is the case in NLP applications in general, we begin by turning each input word into a vector using an embedding algorithm.

<img src="https://user-images.githubusercontent.com/12748752/168201541-73b96f67-a6b5-4b72-9201-4a26dfd7670a.png" width=80%/>

#####  <i><ins> Each word is embedded into a vector of size 512. We'll represent those vectors with these simple boxes</ins></i>

The embedding only happens in the **bottom-most encoder**. The abstraction that is common to all the encoders is that they receive a list of vectors each of the size 512 – In the bottom encoder that would be the word embeddings, but in other encoders, it would be the output of the encoder that’s directly below. The size of this list is hyperparameter we can set – basically it would be the length of the longest sentence in our training dataset.

After embedding the words in our input sequence, each of them flows through each of the two layers of the encoder.

<img src="https://user-images.githubusercontent.com/12748752/168204497-97f950e0-ad92-4037-a076-3eaf07196dcb.png" width=80% />

Here we begin to see one key property of the Transformer, which is that the word in each position flows through its own path in the encoder. There are dependencies between these paths in the self-attention layer. The feed-forward layer does not have those dependencies, however, and thus the various paths can be executed in parallel while flowing through the feed-forward layer.

Next, we’ll switch up the example to a shorter sentence and we’ll look at what happens in each sub-layer of the encoder.

### 🔲 Now We’re Encoding!
As we’ve mentioned already, an encoder receives a list of vectors as input. It processes this list by passing these vectors into a ‘self-attention’ layer, then into a feed-forward neural network, then sends out the output upwards to the next encoder.

> #### The word at each position passes through a **self-attention** process. Then, they each pass through a feed-forward neural network -- the exact same network with each vector flowing through it separately.

![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
## ⬛ Self-Attention at a High Level
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
" Attention is All You Need" this paper first showed us the concept of “self-attention”.
#### Input sentence : "<img src="https://latex.codecogs.com/svg.image?{\color{Purple}&space;\textbf{\textrm&space;{The&space;animal&space;didn't&space;cross&space;the&space;street&space;because&space;it&space;was&space;too&space;tired}}}" title="https://latex.codecogs.com/svg.image?{\color{Purple} \textbf{\textrm {The animal didn't cross the street because it was too tired}}}" align="center" /> "

![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)








## The Transformer Architecture
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
In a groundbreaking 2017 paper, a team of Google researchers suggested that “Attention Is All You Need.” They managed to create an architecture called the Transformer, which significantly improved the state of the art in NMT **_without using any recurrent or convolutional layers_**, just **attention mechanisms** (plus embedding layers, dense layers, normalization layers, and a few other bits and pieces). As an extra bonus, this architecture was also much faster to train and easier to parallelize, so they managed to train it at a fraction of the time and cost of the previous state-of-the-art models.

<img src="https://user-images.githubusercontent.com/12748752/164050988-292430e3-b184-4942-a92e-f2297b1541d1.png" width=50%/>
<ins><b><i> The Transformer architecture</i></b></ins>

Let’s walk through this figure:
* The lefthand part is the encoder. Just like Encoder–Decoder network, it takes as input a batch of sentences represented as sequences of word IDs (the input shape is [batch size, max input sentence length]), and it encodes each word into a 512-dimensional representation (so the encoder’s output shape is [batch size, max input sentence length, 512]). Note that the top part of the encoder is stacked N times (in the paper, N = 6).

* The righthand part is the decoder. During training, it takes the target sentence as input (also represented as a sequence of word IDs), shifted one time step to the right (i.e., a start-of-sequence token is inserted at the beginning). It also receives the outputs of the encoder (i.e., the arrows coming from the left side). Note that the top part of the decoder is also stacked N times, and the encoder stack’s final outputs are fed to the decoder at each of these N levels. Just like earlier, the decoder outputs a probability for each possible next word, at each time step (its output shape is [batch size, max output sentence length, vocabulary length]).

## References:
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition by Aurélien Géron**
* [Jay Alammar](http://jalammar.github.io/illustrated-transformer/)
* [Kheirie Elhariri](https://towardsdatascience.com/attention-is-all-you-need-e498378552f9)
