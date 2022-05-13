## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

## Sequence-to-sequence models  
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* A sequence-to-sequence model is a model that takes a sequence of items (words, letters, features of an images…etc) and outputs another sequence of items.
* Sequence-to-sequence learning (Seq2Seq) is about training models to convert sequences from one domain (e.g. sentences in English) to sequences in another domain (e.g. the same sentences translated to French).
```Python
"the cat sat on the mat" -> [Seq2Seq model] -> "le chat etait assis sur le tapis"
```
* These are deep learning models that have achieved a lot of success in tasks like `machine translation`, `text summarization` and `image captioning`.
* Google Translate started using such a model in production in late 2016. 
* In neural machine translation a sequence is a series of words, processed one after another. The output is, likewise, a series of words:
<img src="https://user-images.githubusercontent.com/12748752/160001494-3df9a3c0-a5e8-46f9-9860-ca354cca3e1f.png" width=60% />

### The Mechanisam
* Under the hood, the model is composed of an **encoder** and a **decoder**.
* The encoder processes each item in the input sequence, it compiles the information it captures into a vector (called the **context**). 
* After processing the entire input sequence, the encoder sends the context over to the decoder, which begins producing the output sequence item by item.
* The context is a vector (an array of numbers, basically) in the case of machine translation. 
* The encoder and decoder tend to both be RNN, LSTM, GRU
<img src="https://user-images.githubusercontent.com/12748752/168195356-8a08298c-9157-4656-9464-0dd4f7d56145.png" />

* In a encoder model placeholders can be one type or combination of multiple types of RNNs.
* **Context Vector:** You can set the size of the context vector when you set up your model. 
* It is basically the number of hidden units in the encoder RNN/LSTM/GRU. 
* The context vector would be of a size like 256, 512, or 1024.
* And the datatype of the Context Vector is float mainly 64bit.


## Encoder
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* The encoder transforms an input sequence of variable length into a fixed-shape context variable **_C_** and encodes the input sequence information in this context variable. 

* Feature extraction and Aggrigation
* Temporal Information - 
   * `The Dog is Barking` `is` is after `Dog` and before `Barking`
* By design, a RNN takes two inputs at each time step: **an input** (in the case of the encoder, one word from the input sentence), and **a hidden state**. 
* The word, however, needs to be represented by a vector. To transform a word into a vector, we turn to the class of methods called “word embedding” algorithms. These turn words into vector spaces that capture a lot of the meaning/semantic information of the words (e.g. king - man + woman = queen).
* Basically the last hidden state generated in the encoder is the Context Vector.
## Decoder
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* The decoder also maintains a hidden state that it passes from one time step to the next. We just didn’t visualize it in this graphic because we’re concerned with the major parts of the model for now.

Let’s now look at another way to visualize a sequence-to-sequence model. This animation will make it easier to understand the static graphics that describe these models. This is called an “unrolled” view where instead of showing the one decoder, we show a copy of it for each time step. This way we can look at the inputs and outputs of each time step.

---
### Encoder–Decoder Network for Neural Machine Translation
#### 1. English sentences to French:
In short, the English sentences are fed to the **encoder**, and the **decoder** outputs the French translations. Note that the French translations are also used as inputs to the decoder, but shifted back by one step. In other words, the decoder is given as input the word that it should have output at the previous step (regardless of what it actually output). For the very first word, it is given the **start-of-sequence (SOS)** token. The decoder is expected to end the sentence with an **end-of-sequence (EOS)** token.

English sentences are **reversed** before they are fed to the encoder. 
* For example, **“I drink milk”** is reversed to **“milk drink I”**. 

This ensures that the beginning of the English sentence will be fed last to the 10 encoder, which is useful because that’s generally the first thing that the decoder needs to translate. 

Each word is initially represented by its ID (e.g., 288 for the word “milk”). Next, an embedding layer returns the word embedding. These word embeddings are what is actually fed to the encoder and the decoder.

<img src="https://user-images.githubusercontent.com/12748752/161118995-bfe94e09-a72a-4f2e-9ebd-a318efc94a6b.png" width=60% />

#### Step 1:
* Each word is initially represented by its ID (e.g., 288 for the word “**milk**”).
* Next, an embedding layer returns the word embedding. These word embeddings are what is actually fed to the encoder and the decoder.

#### Step 2:
* At each step, the decoder outputs a _score_ for each word in the output vocabulary (i.e., French), and then the `softmax` layer turns these scores into probabilities. 
  * **For example**, at the first step the word “**Je**” may have a probability of 20%, “**Tu**” may have a probability of 1%, and so on. The word with the highest probability is output. 
* This is very much like a regular classification task, so you can train the model using the "**_sparse_categorical_crossentropy_**" loss.
> #### Note: At inference time (after training), you will not have the target sentence to feed to the decoder. Instead, simply feed the decoder the word that it output at the previous step, as shown in the following Figure (this will require an embedding lookup that is not shown in the diagram).

<img src="https://user-images.githubusercontent.com/12748752/161141254-2acb3050-5584-486c-82a0-1f107d1d069e.png" width=50% />

## References 
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* [Jay Alammar- "Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)"](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
