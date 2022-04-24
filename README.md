## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
### ◼️ _Neural-Network_
### ◼️ _Perceptron_
* [The Perceptron](https://github.com/iAmKankan/Neural-Network/blob/main/perceptron/README.md)
* [Multi-Layer Perceptron](https://github.com/iAmKankan/Neural-Network/blob/main/perceptron/README.md)
* [Training a Neural Network](https://github.com/iAmKankan/Neural-Network/blob/main/perceptron/README.md#training-perceptron)
### ◼️ _Neural-Network Hyperparameters_
   * [Activation Function](https://github.com/iAmKankan/Neural-Network/blob/main/activation_functions/README.md)
   * [Optimizers](https://github.com/iAmKankan/Neural-Network/tree/main/optimizer#readme)
### ◼️ _RNN, LSTM, GRU_
* [Recurrent Neural Networks](https://github.com/iAmKankan/Neural-Network/blob/main/rnn/README.md)
* Types of Recurrent Neural Networks
  * Simple RNN
  * LSTM
  * GRU
* Oparetions on RNN
   * [Sentiment Analysis](https://github.com/iAmKankan/Neural-Network/blob/main/sentiment.md)
 * Bidirectional Recurrent Neural Networks
### ◼️ _Natural Language Processing(NLP) with Deep Learning_
  * [Sequence to sequence models](https://github.com/iAmKankan/Neural-Network/tree/main/NLP#readme)
  * Encoder–Decoder Network for Neural Machine Translation
### ◼️ _Attention Mechanism_
   * [Attention](https://github.com/iAmKankan/Neural-Network/tree/main/Attention-Mechanisms#readme)
### ◼️ _Neural-Network Common Terms_
  * [Co-occurrence Matrix](#co-occurrence-matrix)
  * [Negative sampling](#negative-sampling)
##  Neural-Network _Common Terms_
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
### _Co-occurrence Matrix_
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* Generally speaking, a co-occurrence matrix will have specific entities in rows (**ER**) and columns (**EC**). 
* The purpose of this matrix is to present the number of times each **ER** appears in the same context as each **EC**. 
* As a consequence, in order to use a co-occurrence matrix, you have to define your entites and the context in which they co-occur.
* In NLP, the most classic approach is to define each entity (ie, lines and columns) as a word present in a text, and the context as a sentence.

<img src="http://latex.codecogs.com/svg.latex?\begin{matrix}&space;&\mathrm{aardvark}&&space;\mathrm{\cdots}&space;&&space;\mathrm{computer}&space;&&space;\mathrm{data}&space;&&space;\mathrm{result}&space;&\mathrm{&space;pie}&&space;\mathrm{sugar}&space;\\\mathrm{cherry}&space;&&space;0&space;&\cdots&&space;2&space;&&space;8&space;&&space;9&&space;442&&space;25\\\mathrm{strawberry}&&space;0&space;&\cdots&&space;0&space;&&space;0&space;&&space;1&space;&&space;60&&space;19\\\mathrm{digital}&&space;{\color{Red}&space;0}&space;&{\color{Red}\cdots}&&space;{\color{Red}1670}&space;&&space;{\color{Red}1683}&space;&&space;{\color{Red}85}&&space;{\color{Red}&space;5}&&space;{\color{Red}4}\\\mathrm{information}&&space;0&space;&\cdots&&space;3325&&space;3982&&space;378&&space;5&&space;13\\\end{matrix}&space;" title="http://latex.codecogs.com/svg.latex?\begin{matrix} &\mathrm{aardvark}& \mathrm{\cdots} & \mathrm{computer} & \mathrm{data} & \mathrm{result} &\mathrm{ pie}& \mathrm{sugar} \\\mathrm{cherry} & 0 &\cdots& 2 & 8 & 9& 442& 25\\\mathrm{strawberry}& 0 &\cdots& 0 & 0 & 1 & 60& 19\\\mathrm{digital}& {\color{Red} 0} &{\color{Red}\cdots}& {\color{Red}1670} & {\color{Red}1683} & {\color{Red}85}& {\color{Red} 5}& {\color{Red}4}\\\mathrm{information}& 0 &\cdots& 3325& 3982& 378& 5& 13\\\end{matrix} " />

#### What are they used for in NLP ?
* The most evident use of these matrix is their ability to provide links between notions. 
* Let's suppose you're working on products reviews. 
* Let's also suppose for simplicity that each review is only composed of short sentences. 
* You'll have something like that :
```
ProductX is amazing.
I hate productY.
```
* Representing these reviews as one co-occurrence matrix will enable you associate products with appreciations.

### Coreference Resolution
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* _Coreference resolution is the task of finding all expressions that refer to the same entity in a text._ 
* It is an important step for a lot of higher level NLP tasks that involve natural language understanding such as document summarization, question answering, and information extraction.
<img src="https://user-images.githubusercontent.com/12748752/161392419-c0339364-baa1-4600-ac6b-4c486964e213.png" width=40%/>

### Context
* Meaning at any point of a sequence.

## Bibliography
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition by Aurélien Géron**
