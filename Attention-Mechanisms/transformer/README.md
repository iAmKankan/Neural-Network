## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

## ⬛ Transformer
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

### 🔲 Types of Transformers:
<img src="https://user-images.githubusercontent.com/12748752/167986704-ca5cb1fe-7730-4b61-a9f5-1aee7dbaa9e9.png" width=70%/>

### 🔲 As a _Blackbox_
In a machine translation application, it would take a sentence in one language, and output its translation in another.
<img src="https://user-images.githubusercontent.com/12748752/164888116-dfdb9a7f-60c1-4038-9bf6-3f47a133a244.png" width=80%/>

### 🔲 The _Encoder-Decoder_ Blocks
Inside the Transformer box _Encoder-Decoder_ Block and a connection between them.

<img src="https://user-images.githubusercontent.com/12748752/164888115-281a74f2-971d-4eb3-8bcb-0bb58b35727b.png" width=80% />

### 🔲 Inside  _Encoder-Decoder_ Stack
The encoding component is a stack of encoders (the paper stacks six of them on top of each other – there’s nothing magical about the number six, one can definitely experiment with other arrangements). The decoding component is a stack of decoders of the same number.
<img src="https://user-images.githubusercontent.com/12748752/167968727-488848ff-40d1-49a9-99ad-61287bebba3e.png" width=80%/>

### 🔲 The individual _Encoder-Decoder_ Blocks
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

## ⬛ Self-Attention at a High Level
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
" Attention is All You Need" this paper first showed us the concept of “self-attention”.
#### Input sentence : "<img src="https://latex.codecogs.com/svg.image?{\color{Purple}&space;\textbf{\textrm&space;{The&space;animal&space;didn't&space;cross&space;the&space;street&space;because&space;it&space;was&space;too&space;tired}}}" title="https://latex.codecogs.com/svg.image?{\color{Purple} \textbf{\textrm {The animal didn't cross the street because it was too tired}}}" align="center" /> "





















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
* [Jay Alammar](http://jalammar.github.io/illustrated-transformer/)
* [Kheirie Elhariri](https://towardsdatascience.com/attention-is-all-you-need-e498378552f9)
