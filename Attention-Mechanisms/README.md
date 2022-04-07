## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

### _Attention Mechanisms_
* Bahdanau Attention
* Multi-head Attention 
* Self-Attention
* Transformer

## Attention Mechanisms
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
The optic nerve of a primate’s visual system receives massive sensory input, far exceeding what the brain can fully process. Fortunately, not all stimuli are created equal. Focalization and concentration of consciousness have enabled primates to direct attention to objects of interest, such as preys and predators, in the complex visual environment. The ability of paying attention to only a small fraction of the information has evolutionary significance, allowing human beings to live and succeed.
### Attention Cues in Biology
To explain how our attention is deployed in the visual world, a two-component framework has emerged and been pervasive. This idea dates back to William James in the 1890s, who is considered the “father of American psychology”. In this framework, subjects selectively direct the spotlight of attention using both the **_nonvolitional_** cue and **_volitional_** cue.

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
The alignment model takes the encoded hidden states, **_h<sub>i</sub>_** , and the previous decoder output,**_S<sub>t-1</sub>_** , to compute a score, **_e<sub>t,i</sub>_** , that indicates how well the elements of the input sequence align with the current output at position, **_t_**. The alignment model is represented by a function, **_a(._)** , which can be implemented by a feedforward neural network:

**_e<sub>t,i</sub>=a(S<sub>t-1</sub>,h<sub>i</sub>)_**
















## References
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)



