## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

## Bidirectional RNNs
In sequence learning, we assumed that our goal is to model the _next output_ given, in the context of a time series or in the context of a language model. While this is a typical scenario, it is not the only one we might encounter. To illustrate the issue, consider the following three tasks of filling in the blank in a text sequence:
```
I am ___.
I am ___ hungry.
I am ___ hungry, and I can eat half a pig.
```

* Depending on the amount of information available, we might fill in the blanks with very different words such as “**happy**”, “**not**”, and “**very**”. Clearly the end of the phrase (if available) conveys significant information about which word to pick. 
* A sequence model that is incapable of taking advantage of this will perform poorly on related tasks. 
   * For instance, to do well in **named entity recognition** (e.g., to recognize whether “**Green**” refers to “**Mr. Green**” or to the color) longer-range context is equally vital. 
* To get some inspiration for addressing the problem let us take a detour to **_probabilistic graphical models_**.

### Dynamic Programming in Hidden Markov Models
The specific technical details of dynamic programming do not matter for understanding the deep learning models but they help in motivating why one might use deep learning and why one might pick specific architectures.

If we want to solve the problem using probabilistic graphical models we could for instance design a latent variable model as follows. At any time step _t_, we assume that there exists some latent variable h<sub>t </sub> that governs our observed emission x<sub>t</sub>  via P(x<sub>t</sub>|h<sub>t</sub>). Moreover, any transition h<sub>t</sub>;rightarrow h<sub>t+1</sub>  is given by some state transition probability . This probabilistic graphical model is then a hidden Markov model as 

---

## Bidirectional RNNs
At each time step, a _regular recurrent layer_ only looks at _past_ and _present_ inputs before generating its output. In other words, it is “causal,” meaning it cannot look into the _future_. 

This type of **RNN** makes sense when **_forecasting time series_**, but for many NLP tasks, such as **_Neural Machine Translation_**, _it is often preferable to look ahead at the next words before encoding a given word_. 
  * **For example**, consider the phrases **“the Queen of the United Kingdom”**, **“the queen of hearts”**, and **“the queen bee”**: to properly encode the word “**queen**,” you need to look ahead. 

<img src="https://user-images.githubusercontent.com/12748752/166150418-090c082d-c9e1-4f46-9a0e-7c6a1c93a0c2.png" width=50% />

##### Step 1:
To implement this, **run two recurrent layers on the same inputs**, 
   * one reading the words from **left to right** and 
   * the other reading them from **right to left**. 
##### Step 2:
Then simply combine their outputs at each time step, typically by **concatenating** them. This is called a _**bidirectional recurrent layer**_. 

To implement a bidirectional recurrent layer in Keras, wrap a recurrent layer in a `keras.layers.Bidirectional` layer. For example, the following code creates a bidirectional GRU layer
```Python
keras.layers.Bidirectional(keras.layers.GRU(10, return_sequences=True))
```

> #### The Bidirectional layer will create a clone of the GRU layer (but in the reverse direction), and it will run both and concatenate their outputs. So although the GRU layer has 10 units, the Bidirectional layer will output 20 values per time step.


## Bibliography
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition by Aurélien Géron**
