## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

### _Attention Mechanisms_
* Bahdanau Attention
* Multi-head Attention 
* Self-Attention
* Transformer

## Attention Mechanisms
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

> #### The ability of paying attention to only a _small fraction of the information_( such as _preys_ and _predators_) has evolutionary significance, allowing human beings to live and succeed.

### Attention Cues in Biology
To explain how our attention is deployed in the visual world, a two-component framework has emerged and been pervasive. This idea dates back to William James in the 1890s, who is considered the “father of American psychology”. In this framework, subjects selectively direct the spotlight of attention using both the **_nonvolitional cue_** and **_volitional cue_**.

* **_Nonvolitional cue_**: Using the nonvolitional cue based on _saliency_ (red cup, non-paper), attention is **involuntarily** directed to the coffee.
* **_Volitional cue_**: Using the volitional cue (want to read a book) that is task-dependent, attention is **directed** to the book under _volitional control_.

### Queries, Keys, and Values
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
Inspired by the nonvolitional and volitional attention cues that explain the attentional deployment, 

In the following we will describe a framework for designing attention mechanisms by incorporating **_nonvolitional cue_** and **_volitional cue_** attention cues.

To begin with, consider the simpler case where only nonvolitional cues are available. To bias selection over sensory inputs, we can simply use a parameterized fully-connected layer or even non-parameterized max or average pooling.

Therefore, what sets attention mechanisms apart from those **fully-connected layers** or **pooling layers** is the inclusion of the volitional cues. 

**_Queries:_**
In the context of attention mechanisms, we refer to **volitional cues** as **queries**. 

**_Values:_**
Given any query, attention mechanisms bias selection over sensory inputs (e.g., intermediate feature representations) via attention pooling. These sensory inputs are called **values** in the context of attention mechanisms. 

**_Keys:_**
More generally, every **value** is paired with a **key**, which can be thought of the nonvolitional cue of that sensory input. 

As shown in the following picture, we can design attention pooling so that the given **query** (_volitional cue_) can interact with **keys** (_nonvolitional cues_), which guides bias selection over values (sensory inputs).





### Building blocks of attention
* Followings are most important building blocks of attention.
1) Reweight
2) Normalization
3) Dot product

![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

## Attention
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
The attention mechanism was introduced to improve the performance of the _encoder-decoder_ model for machine translation. 
#### The idea behind the attention mechanism was to permit the _decoder_ to utilize the most relevant parts of the input sequence in a flexible manner, by a weighted combination of all of the _encoded input vectors_, with the most relevant vectors being attributed the highest weights. 

### The Attention Mechanism
* The attention mechanism was introduced by **Bahdanau et al. (2014)**, to address the _bottleneck problem_ that arises with the use of a fixed-length encoding vector, _where the decoder would have limited access to the information provided by the input_. 
* This is thought to become especially problematic for long and/or complex sequences, where the dimensionality of their representation would be forced to be the same as for shorter or simpler sequences.

We had seen that Bahdanau et al.’s attention mechanism is divided into the step-by-step computations of the alignment scores, the weights and the context vector:

#### 1) Alignment scores: 
The alignment model takes the encoded hidden states, **_h<sub>i</sub>_** , and the previous decoder output,  **_S<sub>t-1</sub>_** , to compute a score,   **_e<sub>t,i</sub>_** , that indicates how well the elements of the input sequence align with the current output at position,  **_t_**. The alignment model is represented by a function,  **_a(._)** , which can be implemented by a feedforward neural network:

**e<sub>t,i</sub> = a(S<sub>t-1</sub>,h<sub>i</sub>)**
#### 2) Weights: 
The weights, ***&alpha; <sub>t,1</sub>***  are computed by applying a softmax operation to the previously computed alignment scores:

**&alpha; <sub>t,1</sub> = softmax(e<sub>t,i</sub>)**

#### 3) Context vector: 
A unique context vector, **_c<sub>t</sub>_** , is fed into the decoder at each time step. It is computed by a weighted sum of all,**_T_** , encoder hidden states:
 
<img src="https://latex.codecogs.com/svg.image?\mathbf{c_t=\sum_{i=1}^{T}&space;\alpha_{t,1}h_i}" title="https://latex.codecogs.com/svg.image?\mathbf{c_t=\sum_{i=1}^{T} \alpha_{t,1}h_i}" />

However, the attention mechanism can be re-formulated into a general form that can be applied to any sequence-to-sequence (abbreviated to seq2seq) task, where the information may not necessarily be related in a sequential fashion.

![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

<img src="https://user-images.githubusercontent.com/12748752/163703981-531eea76-73c2-422d-b19e-6bea87ac6787.png" width=60%/>
<i><b><ins>Neural machine translation using an Encoder–Decoder network with an attention model</i></b></ins>


On the left, you have the encoder and the decoder. Instead of just sending the encoder’s final hidden state to the decoder (which is still done, although it is not shown in the figure), we now send all of its outputs to the decoder. At each time step, the decoder’s memory cell computes a weighted sum of all these encoder outputs: this determines which words it will focus on at this step. The weight ***&alpha; <sub>(t,i)</sub>*** is the weight of the ***i<sup>th</sup>*** encoder output at the ***t<sup>th</sup>*** decoder time step. For example, if the weight ***&alpha; <sub>(3,2)</sub>*** is much larger than the weights ***&alpha; <sub>(3,0)</sub>*** and ***&alpha; <sub>(3,1)</sub>*** , then the decoder will pay much more attention to word number 2 (“milk”) than to the other two words, at least at this time step. The rest of the decoder works just like earlier: at each time step the memory cell receives the inputs we just discussed, plus the hidden state from the previous time step, and finally (although it is not represented in the diagram) it receives the target word from the previous time step (or at inference time, the output from the previous time step). 
### _Attention Layer or Alignment Model_
But where do these ***&alpha; <sub>(t,i)</sub>*** weights come from? It’s actually pretty simple: they are generated by a type of small neural network called an **_alignment model_** (or an **_attention layer_**), which is trained jointly with the rest of the Encoder–Decoder model. This alignment model is illustrated on the righthand side of Figure 16-6. It starts with a time-distributed Dense layer with a single neuron, which receives as input all the encoder outputs, concatenated with the decoder’s previous hidden state (e.g., h ). This layer outputs a score (or energy) for each encoder output (e.g., e ): this score measures how well each output is aligned with the decoder’s previous hidden state. Finally, all the scores go through a softmax layer to get a final weight for each encoder output (e.g., α ). All the weights for a given decoder time step add up to 1 (since the softmax layer is not timedistributed). This particular attention mechanism is called Bahdanau attention (named after the paper’s first author). Since it concatenates the encoder output with the decoder’s previous hidden state, it is sometimes called concatenative attention (or additive attention).



### _EXPLAINABILITY:_
One extra benefit of attention mechanisms is that they make it easier to understand what led the model to produce its output. This is called explainability. It can be especially useful when the model makes a mistake: for example, if an image of a dog walking in the snow is labeled as “a wolf walking in the snow,” then you can go back and check what the model focused on when it output the word “wolf.” You may find that it was paying attention not only to the dog, but also to the snow, hinting at a possible explanation: perhaps the way the model learned to distinguish dogs from wolves is by checking whether or not there’s a lot of snow around. You can then fix this by training the model with more images of wolves without snow, and dogs with snow. This example comes from a great 2016 paper by Marco Tulio Ribeiro et al. that uses a different approach to explainability: learning an interpretable model locally around a classifier’s prediction. In some applications, explainability is not just a tool to debug a model; it can be a legal requirement (think of a system deciding whether or not it should grant you a loan).


## References
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)



