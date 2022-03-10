## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
### Neural-Network
* [The Perceptron](https://github.com/iAmKankan/Neural-Network/blob/main/perceptron.md)
* [Multi-Layer Perceptron](https://github.com/iAmKankan/Neural-Network/blob/main/multilayer-perceptrons.md)
* [Training a Neural Network](https://github.com/iAmKankan/Neural-Network/blob/main/training.md)
   * [Activation Function](https://github.com/iAmKankan/Neural-Network/tree/main/activation_functions#readme)
   * [Optimizers](https://github.com/iAmKankan/Neural-Network/tree/main/optimizer#readme)
* [Recurrent Neural Networks](https://github.com/iAmKankan/Neural-Network/blob/main/rnn.md)
* Types of Recurrent Neural Networks
  * Simple RNN
  * LSTM
  * GRU
* Oparetions on RNN
   * [Sentiment Analysis](https://github.com/iAmKankan/Neural-Network/blob/main/sentiment.md)

### Neural-Network _Common Terms_
  * [Co-occurrence Matrix](#co-occurrence-matrix)
  * [Negative sampling](#negative-sampling)

## Neural-Network _Common Terms_
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
### _Negative sampling_
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* Negative sampling reduces computation by sampling just **N** negative instances along with the target word instead of sampling the whole vocabulary.
* Technically, negative sampling ignores most of the ‘0’ in the one-hot label word vector, and only propagates and updates the weights for the target and a few negative classes which were randomly sampled.
* More concretely, negative sampling samples negative instances(words) along with the target word and minimizes the log-likelihood of the sampled negative instances while maximizing the log-likelihood of the target word.
> ##### Samples very near to 0 is treated as negative (-ve) and samples are very far from 0 is treated as positive(+ve).
<img src="https://user-images.githubusercontent.com/12748752/157604235-9119cfe7-eb3f-48dd-bb0c-fcf03283af6a.png" width=50% />

#### Sub-sampling <img src="https://latex.codecogs.com/svg.image?{\color{Red}&space;\textbf{Depricated}}" align="center">
Some frequent words often provide little information. Words with frequency above a certain threshold (e.g ‘a’, ‘an’ and ‘that’) may be subsampled to increase training speed and performance. Also, common word pairs or phrases may be treated as single “words” to increase training speed.

#### Context window
* The size of the context window determines how many words before and after a given word would be included as context words of the given word. According to the authors’ note, the recommended value is 10 for skip-gram and 5 for CBOW.
* Here is an example of Skip-Gram with context window of size 2:
<img src="https://user-images.githubusercontent.com/12748752/157604214-1127794b-9dc7-40cf-8249-f2e453441d05.png" width=50% />

## Bibliography
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition by Aurélien Géron**
