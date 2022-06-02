## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

## ⬛ Transformer
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
The architecture of the **_transformer_** model inspires from the **attention mechanism** used in the **encoder-decoder** architecture in **RNNs** to handle s**equence-to-sequence (seq2seq)** tasks.

Unlike [**self-attention models**](https://github.com/iAmKankan/Neural-Network/blob/main/Attention-Mechanisms/self-attention.md#-self-attention-and-positional-encoding),  that still rely on _RNNs_ for **_input representations_**, the **transformer model** is solely based on **_attention mechanisms_** without any **_convolutional_** or **_recurrent_** layer. 

Transformer eliminates the factor of **sequentiality**; meaning that, unlike _RNNs_, _the transformer does not process data in_ **_sequence (i.e. in order)_**, which allows for **more** [**_parallelization_**](https://github.com/iAmKankan/MachineLearning_With_Python/blob/master/README.md#parallelization) and **reduces** [**_training time_**](https://github.com/iAmKankan/Data-Structure/blob/main/complexity.md).

Though originally proposed for **sequence to sequence learning** on **text data**, _transformers_ have been pervasive in a wide range of modern deep learning applications, such as in areas of **language**, **vision**, **speech**, and **reinforcement learning**.

### 🔲 Classification of Transformers:
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
<img src="https://user-images.githubusercontent.com/12748752/167986704-ca5cb1fe-7730-4b61-a9f5-1aee7dbaa9e9.png" width=70%/>


### 🔲 The Model
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
As we can see, the **transformer** is composed of an **_encoder_** and a **_decoder_**. Different from [Bahdanau attention](https://github.com/iAmKankan/Neural-Network/blob/main/Attention-Mechanisms/bahdanau_attention.md) for _sequence to sequence learning_, the **input** (_source_) and **output** (_target_) **sequence embeddings** are added with [**positional encoding**](https://github.com/iAmKankan/Neural-Network/blob/main/Attention-Mechanisms/self-attention.md#-positional-encoding) before being fed into the **encoder** and the **decoder** that stack modules based on **self-attention**.

<img src="https://user-images.githubusercontent.com/12748752/164050988-292430e3-b184-4942-a92e-f2297b1541d1.png" width=30%/>
<p align="center"><ins><i>The transformer architecture</i></ins>.</p>

## The architecture in detail:
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
Here, the classical example of translating from _English to French_ using the transformer is considered. Input sentence is as such <img src="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{I\&space;am\&space;a\&space;student}&space;}&space;&space;" title="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{I\ am\ a\ student} } " />, and the expected output is <img src="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{Je\&space;suis\&space;un\&space;\acute{e}tudiant}&space;}&space;" title="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{Je\ suis\ un\ \acute{e}tudiant} }" />. 

In a _machine translation_ application, it would take a sentence in one language( here is English), and output its translation in another(here is French).

### <ins>Transformers as a _Blackbox_</ins>
<img src="https://user-images.githubusercontent.com/12748752/164888116-dfdb9a7f-60c1-4038-9bf6-3f47a133a244.png" width=80%/>

### <ins>The Encoder-Decoder Blocks</ins>
* Inside the Transformer box, exist _Encoder-Decoder_ Blocks and a connection between them.
* The **Encoding components** are in the stack of encoders (the paper stacks **six** of them on top of each other – there’s nothing magical about the number **6**, one can definitely experiment with other arrangements). The decoding components are in the stack of decoders of the _same number_(**6**).

<img src="https://user-images.githubusercontent.com/12748752/164888115-281a74f2-971d-4eb3-8bcb-0bb58b35727b.png" width=80% align="center"/> 

### <ins>Inside each Encoder-Decoder Stack</ins>
<img src="https://user-images.githubusercontent.com/12748752/167968727-488848ff-40d1-49a9-99ad-61287bebba3e.png" width=80% align="center"/>


### <ins>Inside each Encoder-Decoder Block</ins>
<img src="https://user-images.githubusercontent.com/12748752/168034980-004fd235-28cb-4831-9523-76480b411e11.png" width=80% align="center"/> 


### 🔲 The Encoder
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
<img src="https://user-images.githubusercontent.com/12748752/171049973-6959aa04-a62b-4a5c-abbe-f2481462ea74.png" width=30%/>
<p align="center"><ins><i>A single Encoder Block</i></ins></p>

* The **_transformer encoder_** is a stack of _multiple identical layers_ with **_residual connections_** and **_layer normalizations_**, where each layer has **two sublayers** (_either is denoted as sublayer_).
    * The first is a **multi-head self-attention pooling** and 
    * The second is a **positionwise feed-forward network**. 
* Specifically, in the encoder **self-attention**- **queries**, **keys** and **values** are all from the the _outputs_ of the previous encoder layer. 
* Inspired by the **ResNet** design, a **residual connection** is employed around **both sublayers**. 
* In the transformer, for any input <img src="https://latex.codecogs.com/gif.image?\dpi{110}{\color{Purple}&space;\mathbf{x&space;\in&space;\mathbb{R}^{d}}}&space;" title="https://latex.codecogs.com/gif.image?\dpi{110}{\color{Purple} \mathbf{x \in \mathbb{R}^{d}}} " align="center" /> at any position of the sequence, we require that <img src="https://latex.codecogs.com/gif.image?\dpi{110}{\color{Purple}&space;\mathbf{sublayer(x)&space;\in&space;\mathbb{R}^{d}}}&space;" title="https://latex.codecogs.com/gif.image?\dpi{110}{\color{Purple} \mathbf{sublayer(x) \in \mathbb{R}^{d}}} " align="center"/> so that the residual connection <img src="https://latex.codecogs.com/gif.image?\dpi{110}{\color{Purple}&space;\mathbf{x&plus;sublayer(x)&space;\in&space;\mathbb{R}^{d}}}&space;" title="https://latex.codecogs.com/gif.image?\dpi{110}{\color{Purple} \mathbf{x+sublayer(x) \in \mathbb{R}^{d}}} " align="center" /> is feasible. 
* This addition from the _residual connection_ is immediately followed by **_layer normalization_**. 
* As a result, the **transformer encoder** outputs a _d-dimensional vector_ representation for _each position_ of the input sequence.

### <ins>_The input_</ins>
The raw data is an english text( in this particular scenario), however the transformer, like any other model, does not understand english language and thus, _the text is processed to convert every word into a_ **_unique numeric ID_**. 

This is done by using a specific dictionary of vocabulary, which can be generated from the training data, and that maps each word to a **numeric index**.

### <ins>_Embedding Layer_</ins>
As in other models, the transformer uses learned embeddings to transform the **input tokens** into **vectors of dimension d = 512**. During training, the model updates the numbers in the vectors to better represent the input tokens

### <ins>_Positional Encoding_</ins>
* Refer to the [**Positional Encoding** in self-Attention page](https://github.com/iAmKankan/Neural-Network/blob/main/Attention-Mechanisms/self-attention.md#-positional-encoding)
### <ins>_The Multi-Head Attention Layer — Self-Attention_</ins>
An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

<img src="https://user-images.githubusercontent.com/12748752/171082932-d39bfe4b-8ae6-4f93-b6ee-3b8d53e1a4bb.png" width=50%/>

<p align="center"><ins><i><b>(left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.</b></i></ins>.</p>

![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)


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


![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)


### 🔲 The Decoder
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
<img src="https://user-images.githubusercontent.com/12748752/171049969-c7791fe9-5c19-4459-9bca-ca48944c7597.png" width=25%/>
<p align="center"> <ins><i>A single Decoder Block</i></ins></p>

The **_transformer decoder_** is also a stack of _multiple identical layers_ with **_residual connections_** and **_layer normalizations_**, the **decoder** inserts one more sublayer (total **three**), known as the **encoder-decoder attention**, between these two layers. 
* In the **encoder-decoder attention** , **queries** are from the _outputs_ of the _previous decoder layer_, and the **keys** and **values** are from the **transformer encoder outputs**. 
* In the decoder **self-attention**- **queries**, **keys**, and **values** are all from the the outputs of the **previous decoder layer**. 
* However, each position in the decoder is allowed to only attend to all positions in the decoder up to that position.
* This **masked attention** preserves the **auto-regressive** property, ensuring that the _prediction only depends on those output tokens that have been generated_.

We have already described and implemented multi-head attention based on [scaled dot-products](https://github.com/iAmKankan/Neural-Network/blob/main/Attention-Mechanisms/multi-head.md) and [positional encoding](https://github.com/iAmKankan/Neural-Network/blob/main/Attention-Mechanisms/self-attention.md#-positional-encoding).

* The main differences between the decoder and the encoder are that the **decoder** takes in **two inputs** and applies **_multi-head attention_** _twice_ with one of them being "**masked**". 
* Also, the **final linear layer** in the decoder has the size (i.e. number of units) equal to the number of words in the target dictionary (in this case the french language dictionary). Each unit will be assigned a score; the softmax is applied to convert these scores into probabilities indicating the probability of each word to be present in the output.

### <ins>_The input_</ins>
The decoder takes in two inputs:

1. **The output of the encoder** — these are the **keys (K)** and the **values (V)** that the decoder performs **multi-head attention on** . In this **multi-head attention layer**, the **query (Q)** is the output of the masked multi-head attention.
2. **The output text shifted to the right** — This is to ensure that predictions at a specific position **"i"** can only depend at positions less than **i** (see figure below). Therefore, the decoder will take in all words already predicted (position **0 to i-1**) before the actual word to be predicted at position **i**. Note that the first generated word passed to the decoder is the token `<start>` and the prediction process continues until the decoder generates a special end token `<eos>`.

 <img src="https://user-images.githubusercontent.com/12748752/169290757-0d143632-7fd4-45af-857e-c25ee5db6ed9.gif" />

 <ins><i><b>Outputs Shifted by Right as Inputs to the Decoder In the Inference Stage   </b></i></ins>(Image by ['Kheirie Elhariri](https://towardsdatascience.com/attention-is-all-you-need-e498378552f9))
 
 
 ### _◼️ Masked Multi-Head Attention_
The process of the masked multi-head attention is similar to that of the regular multi-head attention. The only difference is that after multiplying the matrices Q and K, and scaling them, a special mask is applied on the resulting matrix before applying the softmax (see left diagram of figure 6-Mask opt.). The objective is to have every word at a specific position "i" in the text to only attend to every other position in the text up until its current position included (position 0 until position i). This is important in the training phase, as when predicting the word at position i+1, the model will only pay attention to all the words before that position. Therefore, all positions after i, are masked and set to negative infinity before passing them to the softmax operation, which results in 0s in the attention filter (see figure 11).
 
 
 ### 🔲 The Conclusion
The Transformer model is a deep learning model that has been in the field for five years now, and that has lead to several top performing and state of the art models such as the BERT model. Giving its dominance in the field of NLP and its expanding usage in other fields such as computer vision, it is important to understand its architecture. This article covers the different components of the transformer and highlights their functionalities.
 
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

### 🔲 The individual _Encoder-Decoder_ Blocks
<img src="https://user-images.githubusercontent.com/12748752/164050988-292430e3-b184-4942-a92e-f2297b1541d1.png" align="right" width=25% />

The encoders are all identical in structure (yet they do not share weights). Each one is broken down into two sub-layers:
#### Each Encoder block having two components
  1) A **Feedforward layer**
  2) A **Self-attention** layer
The encoder’s inputs first flow through a **self-attention layer** – _a layer that helps the encoder look at other words in the input sentence as it encodes a specific word._

The outputs of the **self-attention layer** are fed to a **feed-forward** neural network. The exact same neural network is independently applied to each position.
#### Each Decoder block having three components
  1) A **Feedforward layer** 
  2) A **Self-attention** layer
  3) A **Encoder-Decoder Attention** layer
  
The decoder has both those layers, but between them is an **attention layer** that helps the decoder focus on relevant parts of the input sentence (_similar what attention does in seq2seq models_).


![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

## ⬛ Embedding Algorithm
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

Let’s start to look at the various **vectors**/**tensors** and how they flow between the _above components_ to turn the _input of a trained model into an output_.
<img src="https://user-images.githubusercontent.com/12748752/168201541-73b96f67-a6b5-4b72-9201-4a26dfd7670a.png" width=80%/>

<p align="center"><i><ins><b> Each word is embedded into a vector of size 512. We'll represent those vectors with these simple boxes</b></ins></i></p>

* As is the case in NLP applications in general, we begin by turning each input word into a **vector** using an **_embedding algorithm_**.
* The _embedding only happens_ in the **bottom-most encoder**. 
* It is common to all the **encoders** is that they receive a **list of vectors each of the size 512** – In the bottom encoder that would be the **word embeddings**, but in **other encoders**, it would be the output of the encoder that’s directly below. 
* The size of this list is **hyperparameter** we can set – **_basically it would be the length of the longest sentence in our training dataset_**.
* After embedding the words in our **input sequence**, each of them flows through each of the two layers of the encoder.

<img src="https://user-images.githubusercontent.com/12748752/168204497-97f950e0-ad92-4037-a076-3eaf07196dcb.png" width=80% />

#### _Important Notes_:
* One key property of the **Transformer**, _the word in each position flows through its own path in the encoder_. There are dependencies between these paths in the **self-attention layer**. 
* The **feed-forward layer** does not have those dependencies, how ever and thus the various paths can be executed in parallel while flowing through the feed-forward layer.

Next, we’ll switch up the example to a shorter sentence and we’ll look at what happens in each sub-layer of the encoder.

### 🔲 Now We’re Encoding!
* As we’ve mentioned already, an **encoder** receives a _list of vectors as input_. 
* It processes this list by passing these vectors into a ‘**self-attention**’ layer, then into a **feed-forward neural network**, then sends out the output upwards to the next encoder.

<img src="https://user-images.githubusercontent.com/12748752/171093005-300c9b35-ee50-44f0-9d73-1cefc56067ca.png" width=80%/>

<p align="center"><i><ins><b>The word at each position passes through a self-attention process. Then, they each pass through a feed-forward neural network -- the exact same network with each vector flowing through it separately.</b></ins></i></p>


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


![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

## The Transformer Architecture
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
In a groundbreaking 2017 paper, a team of Google researchers suggested that “Attention Is All You Need.” They managed to create an architecture called the Transformer, which significantly improved the state of the art in NMT **_without using any recurrent or convolutional layers_**, just **attention mechanisms** (plus embedding layers, dense layers, normalization layers, and a few other bits and pieces). As an extra bonus, this architecture was also much faster to train and easier to parallelize, so they managed to train it at a fraction of the time and cost of the previous state-of-the-art models.

<img src="https://user-images.githubusercontent.com/12748752/164050988-292430e3-b184-4942-a92e-f2297b1541d1.png" width=50%/>
<ins><b><i> The Transformer architecture</i></b></ins>

Let’s walk through this figure:
* The lefthand part is the encoder. Just like Encoder–Decoder network, it takes as input a batch of sentences represented as sequences of word IDs (the input shape is [batch size, max input sentence length]), and it encodes each word into a 512-dimensional representation (so the encoder’s output shape is [batch size, max input sentence length, 512]). Note that the top part of the encoder is stacked N times (in the paper, N = 6).

* The righthand part is the decoder. During training, it takes the target sentence as input (also represented as a sequence of word IDs), shifted one time step to the right (i.e., a start-of-sequence token is inserted at the beginning). It also receives the outputs of the encoder (i.e., the arrows coming from the left side). Note that the top part of the decoder is also stacked N times, and the encoder stack’s final outputs are fed to the decoder at each of these N levels. Just like earlier, the decoder outputs a probability for each possible next word, at each time step (its output shape is [batch size, max output sentence length, vocabulary length]).



![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

### ◼️ 1) <ins>_Positionwise Feed-Forward Networks_</ins> 
The **positionwise feed-forward network** transforms the representation at all the sequence positions using the same **Multilayer Perceptron**(MLP). This is why we call it positionwise. In the implementation below, the input **X** with shape (batch size, number of time steps or sequence length in tokens, number of hidden units or feature dimension) will be transformed by a two-layer MLP into an output tensor of shape (batch size, number of time steps, ffn_num_outputs).

```Python
#@save
class PositionWiseFFN(tf.keras.layers.Layer):
    """Positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super().__init__(*kwargs)
        self.dense1 = tf.keras.layers.Dense(ffn_num_hiddens)
        self.relu = tf.keras.layers.ReLU()
        self.dense2 = tf.keras.layers.Dense(ffn_num_outputs)

    def call(self, X):
        return self.dense2(self.relu(self.dense1(X)))
        
ffn = PositionWiseFFN(4, 8)
ffn(tf.ones((2, 3, 4)))[0]
```        

### 2) <ins>_Residual Connection and Layer Normalization_</ins> 
Now let us focus on the “add & norm” component in Fig. 10.7.1. As we described at the beginning of this section, this is a residual connection immediately followed by layer normalization. Both are key to effective deep architectures.

In Section 7.5, we explained how batch normalization recenters and rescales across the examples within a minibatch. Layer normalization is the same as batch normalization except that the former normalizes across the feature dimension. Despite its pervasive applications in computer vision, batch normalization is usually empirically less effective than layer normalization in natural language processing tasks, whose inputs are often variable-length sequences.

The following code snippet compares the normalization across different dimensions by layer normalization and batch normalization.

### Summary
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* The transformer is an instance of the encoder-decoder architecture, though either the encoder or the decoder can be used individually in practice.
* In the transformer, multi-head self-attention is used for representing the input sequence and the output sequence, though the decoder has to preserve the auto-regressive property via a masked version.
* Both the residual connections and the layer normalization in the transformer are important for training a very deep model.
* The positionwise feed-forward network in the transformer model transforms the representation at all the sequence positions using the same MLP.


## References:
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition by Aurélien Géron**
* [Jay Alammar](http://jalammar.github.io/illustrated-transformer/)
* [Kheirie Elhariri](https://towardsdatascience.com/attention-is-all-you-need-e498378552f9)
