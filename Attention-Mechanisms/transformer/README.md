## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

## Transformer
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

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



## The Transformer Architecture
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
In a groundbreaking 2017 paper, a team of Google researchers suggested that “Attention Is All You Need.” They managed to create an architecture called the Transformer, which significantly improved the state of the art in NMT **_without using any recurrent or convolutional layers_**, just **attention mechanisms** (plus embedding layers, dense layers, normalization layers, and a few other bits and pieces). As an extra bonus, this architecture was also much faster to train and easier to parallelize, so they managed to train it at a fraction of the time and cost of the previous state-of-the-art models.

<img src="https://user-images.githubusercontent.com/12748752/164050988-292430e3-b184-4942-a92e-f2297b1541d1.png" width=50%/>
<ins><b><i> The Transformer architecture</i></b></ins>

Let’s walk through this figure:
* The lefthand part is the encoder. Just like Encoder–Decoder network, it takes as input a batch of sentences represented as sequences of word IDs (the input shape is [batch size, max input sentence length]), and it encodes each word into a 512-dimensional representation (so the encoder’s output shape is [batch size, max input sentence length, 512]). Note that the top part of the encoder is stacked N times (in the paper, N = 6).

* The righthand part is the decoder. During training, it takes the target sentence as input (also represented as a sequence of word IDs), shifted one time step to the right (i.e., a start-of-sequence token is inserted at the beginning). It also receives the outputs of the encoder (i.e., the arrows coming from the left side). Note that the top part of the decoder is also stacked N times, and the encoder stack’s final outputs are fed to the decoder at each of these N levels. Just like earlier, the decoder outputs a probability for each possible next word, at each time step (its output shape is [batch size, max output sentence length, vocabulary length]).
