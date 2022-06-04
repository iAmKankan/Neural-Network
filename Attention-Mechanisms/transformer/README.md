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

## 🔲 The architecture in brief:
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

### Embedding
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

Let’s start to look at the various **vectors**/**tensors** and how they flow between the _above components_ to turn the _input of a trained model into an output_.

<img src="https://user-images.githubusercontent.com/12748752/168201541-73b96f67-a6b5-4b72-9201-4a26dfd7670a.png" width=80%/>

<p align="center"><i><ins><b> Each word is embedded into a vector of size 512. We'll represent those vectors with these simple boxes</b></ins></i></p>

* As is the case in NLP applications in general, we begin by turning each input word into a **vector** using an **_embedding algorithm_**.
* The _embedding only happens_ in the **bottom-most encoder**. 
* It is common to all the **encoders** is that they receive a **list of vectors each of the size 512** – In the bottom encoder that would be the **word embeddings**, but in **other encoders**, it would be the output of the encoder that’s directly below. 
* The size of this list is **hyperparameter** we can set – **_basically it would be the length of the longest sentence in our training dataset_**.
* After embedding the words in our **input sequence**, each of them flows through each of the two layers of the encoder.

<img src="https://user-images.githubusercontent.com/12748752/168204497-97f950e0-ad92-4037-a076-3eaf07196dcb.png" width=80% />

### <ins>_Important Notes_</ins>:
* One key property of the **Transformer**, _the word in each position flows through its own path in the encoder_. There are dependencies between these paths in the **self-attention layer**. 
* The **feed-forward layer** does not have those dependencies, how ever and thus the various paths can be executed in parallel while flowing through the feed-forward layer.


## 🔲 The Encoder
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

<img src="https://user-images.githubusercontent.com/12748752/171049973-6959aa04-a62b-4a5c-abbe-f2481462ea74.png" width=30%/>
<p align="center"><ins><i><b>A single Encoder Block</b></i></ins></p>

* As we’ve mentioned already, an **encoder** receives a _list of vectors as input_. 
* It processes this list by passing these vectors into a ‘**self-attention**’ layer, then into a **feed-forward neural network**, then sends out the output upwards to the next encoder.

<img src="https://user-images.githubusercontent.com/12748752/171093005-300c9b35-ee50-44f0-9d73-1cefc56067ca.png" width=80%/>

<p align="center"><i><ins><b>The word at each position passes through a self-attention process. Then, they each pass through a feed-forward neural network -- the exact same network with each vector flowing through it separately.</b></ins></i></p>

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
* [In detail](https://github.com/iAmKankan/Neural-Network/blob/main/Attention-Mechanisms/transformer/README.md#embedding)

### <ins>_Self-Attention_</ins>
* [Overview - Self-Attention](https://github.com/iAmKankan/Neural-Network/blob/main/Attention-Mechanisms/self-attention.md#-self-attention-at-a-high-level)
* Self-Attention in detail
    * [Vector Calculation - Self-Attention](https://github.com/iAmKankan/Neural-Network/blob/main/Attention-Mechanisms/self-attention.md#self-attention-using-vectors)
    * [Matrix Calculation - Self-Attention](https://github.com/iAmKankan/Neural-Network/blob/main/Attention-Mechanisms/self-attention.md#matrix-calculation-of-self-attention)

### <ins>_Multi-Headed Attention_</ins>
   * [Multi-Headed Attention in detail](https://github.com/iAmKankan/Neural-Network/blob/main/Attention-Mechanisms/self-attention.md#multi-headed-attention) 

### <ins>_Positional Encoding_</ins>
* [How Transformer approach differs from Sequence Model approach:](https://github.com/iAmKankan/Neural-Network/blob/main/Attention-Mechanisms/p-encoding_residual.md#how-transformer-approach-differs-from-sequence-model-approach)
   * Problems associated with this approach
   * Solution
* [A fixed positional encoding based on sine and cosine functions:](https://github.com/iAmKankan/Neural-Network/blob/main/Attention-Mechanisms/p-encoding_residual.md#a-fixed-positional-encoding-based-on-sine-and-cosine-functions)

### <ins>_Residual connection_</ins>
* [The Residuals](https://github.com/iAmKankan/Neural-Network/blob/main/Attention-Mechanisms/p-encoding_residual.md#-the-residuals)


### <ins>_Layer-Normalization_</ins>
* [Crude version](https://arxiv.org/abs/1607.06450)

### 🔲 The Decoder
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
<img src="https://user-images.githubusercontent.com/12748752/171049969-c7791fe9-5c19-4459-9bca-ca48944c7597.png" width=25%/>
<p align="center"> <ins><i>A single Decoder Block</i></ins></p>

Most of the concepts has been covered on the **encoder** side, we basically know how the components of **decoders** work as well. But let’s take a look at how they work together.

* The **encoder** start by processing the **input sequence**. 
* The **output** of the **top encoder** is then transformed into a set of attention vectors **K** and **V**. 
* These are to be used by each **decoder** in its “**_encoder-decoder attention_**” layer which helps the **decoder** focus on appropriate places in the input sequence:

<img src="https://user-images.githubusercontent.com/12748752/171801079-2b6bd472-c68c-4a76-b212-a0c97c9313b5.gif" width=50%/>

<p align="center"> <ins><i>After finishing the encoding phase, we begin the decoding phase. Each step in the decoding phase outputs an element from the output sequence (the English translation sentence in this case).</i></ins></p>

The following steps repeat the process until a special symbol is reached indicating the transformer decoder has completed its output. The output of each step is fed to the bottom decoder in the next time step, and the decoders bubble up their decoding results just like the encoders did. And just like we did with the encoder inputs, we embed and add positional encoding to those decoder inputs to indicate the position of each word.

<img src="https://user-images.githubusercontent.com/12748752/171801041-4259d732-a49b-4e89-9506-1fa1d2d4cf4b.gif" width=50%/>


The self attention layers in the decoder operate in a slightly different way than the one in the encoder:

In the decoder, the self-attention layer is only allowed to attend to earlier positions in the output sequence. This is done by masking future positions (setting them to -inf) before the softmax step in the self-attention calculation.

The “Encoder-Decoder Attention” layer works just like multiheaded self-attention, except it creates its Queries matrix from the layer below it, and takes the Keys and Values matrix from the output of the encoder stack.



## The Final Linear and Softmax Layer
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
The decoder stack outputs a vector of floats. How do we turn that into a word? That’s the job of the final Linear layer which is followed by a Softmax Layer.

The Linear layer is a simple fully connected neural network that projects the vector produced by the stack of decoders, into a much, much larger vector called a logits vector.

Let’s assume that our model knows 10,000 unique English words (our model’s “output vocabulary”) that it’s learned from its training dataset. This would make the logits vector 10,000 cells wide – each cell corresponding to the score of a unique word. That is how we interpret the output of the model followed by the Linear layer.

The softmax layer then turns those scores into probabilities (all positive, all add up to 1.0). The cell with the highest probability is chosen, and the word associated with it is produced as the output for this time step.


<img src="https://user-images.githubusercontent.com/12748752/171743551-8454433a-4885-4a79-afc3-c1b23d5322d9.png" width=50%/>
<p align="center"> <ins><i><b>This figure starts from the bottom with the vector produced as the output of the decoder stack. It is then turned into an output word.</b></i></ins></p>




![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)


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
