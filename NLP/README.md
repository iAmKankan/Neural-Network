## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

## Sequence-to-sequence models  
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* A sequence-to-sequence model is a model that takes a sequence of items (words, letters, features of an images…etc) and outputs another sequence of items.
* These are deep learning models that have achieved a lot of success in tasks like `machine translation`, `text summarization` and `image captioning`.
* Google Translate started using such a model in production in late 2016. 
* In neural machine translation a sequence is a series of words, processed one after another. The output is, likewise, a series of words:

### The Mechanisam
* Under the hood, the model is composed of an **encoder** and a **decoder**.
* The encoder processes each item in the input sequence, it compiles the information it captures into a vector (called the **context**). 
* After processing the entire input sequence, the encoder sends the context over to the decoder, which begins producing the output sequence item by item.
* The context is a vector (an array of numbers, basically) in the case of machine translation. 
* The encoder and decoder tend to both be RNN, LSTM, GRU
<img src="https://user-images.githubusercontent.com/12748752/159683212-c666dd34-a293-4b7e-881f-7d60bef23663.png" />

* In a encoder model placeholders can be one type or combination of multiple types of RNNs.
* **Context Vector:** You can set the size of the context vector when you set up your model. 
* It is basically the number of hidden units in the encoder RNN/LSTM/GRU. 
* The context vector would be of a size like 256, 512, or 1024.
* And the datatype of the Context Vector is float mainly 64bit.

<img src="http://latex.codecogs.com/svg.latex?\begin{matrix}The,&space;&&space;&space;Man,&space;&space;&&space;&space;Loves,&space;&&space;His,&space;&&space;&space;Family&space;&space;\\\end{matrix}" title="http://latex.codecogs.com/svg.latex?\begin{matrix}The, & Man, & Loves, & His, & Family \\\end{matrix}" />


## References 
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* [Jay Alammar- "Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)"](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
