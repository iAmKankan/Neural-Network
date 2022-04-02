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
