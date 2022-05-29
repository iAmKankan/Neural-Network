## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

### _Attention Mechanisms_
* [Bahdanau Attention](https://github.com/iAmKankan/Neural-Network/blob/main/Attention-Mechanisms/bahdanau_attention.md)
* [Multi-head Attention](https://github.com/iAmKankan/Neural-Network/blob/main/Attention-Mechanisms/multi-head.md) 
* [Self-Attention](https://github.com/iAmKankan/Neural-Network/blob/main/Attention-Mechanisms/self-attention.md)
* [Transformer](https://github.com/iAmKankan/Neural-Network/blob/main/Attention-Mechanisms/transformer/README.md)

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

<ins>**_Bias Selection_**</ins>**:** Consider, only **nonvolitional cues** are available. To bias selection over **sensory inputs**, we can simply use a _parameterized fully-connected layer_ or even non-parameterized max or average pooling. Therefore, what sets attention mechanisms apart from those **fully-connected layers** or **pooling layers** is the inclusion of the volitional cues. 

<ins>**_Queries:_**</ins>
In the context of attention mechanisms, we refer to **volitional cues** as **queries**. 

<ins>**_Values:_**</ins>
Given any query, attention mechanisms bias selection over sensory inputs (e.g., intermediate feature representations) via attention pooling. These sensory inputs are called **values** in the context of attention mechanisms. 

<ins>**_Keys:_**</ins>
More generally, every **value** is paired with a **key**, which can be thought of the nonvolitional cue of that sensory input. 

As shown in the following picture, we can design attention pooling so that the given **query** (_volitional cue_) can interact with **keys** (_nonvolitional cues_), which guides bias selection over values (sensory inputs).

<img src="https://user-images.githubusercontent.com/12748752/170866116-ea8da04e-a069-447c-b60d-4d2d8823be33.png" width=70%/>

<ins><i><b>Attention mechanisms bias selection over values (sensory inputs) via attention pooling, which incorporates queries (volitional cues) and keys (nonvolitional cues).</b></i></ins>

## Summary: 
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* Human attention is a **limited**, **valuable**, and **scarce resource**.
* Subjects selectively direct attention using both the **nonvolitional**(based on **saliency**) and **volitional cues**(task-dependent).
* **Attention mechanisms** are different from **fully-connected layers** or **pooling layers** _due to inclusion of the_ **_volitional cues_**.
* **Attention mechanisms** <ins>**_bias_**</ins> selection over **_values_** (_sensory inputs_) via **attention pooling**, which incorporates **queries** (_volitional cues_) and **keys** (_nonvolitional cues_). **Keys** and **values** are paired.
* We can visualize **attention weights** between **queries** and **keys**.


## _Attention Pooling_: Nadaraya-Watson Kernel Regression
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
As we have seen in the above figure the major components of attention mechanisms. 
* The interactions between **queries** (_volitional cues_) and_ **keys** (_nonvolitional cues_) result in **_attention pooling_**. 
* **The attention pooling** selectively aggregates **values** (_sensory inputs_) to produce the output. 

In this section, we will describe attention pooling in greater detail to give you a high-level view of how attention mechanisms work in practice. Specifically, the _Nadaraya-Watson_ kernel regression model proposed in 1964 is a simple yet complete example for demonstrating machine learning with attention mechanisms.

### Generating the Dataset
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

Let us consider the following regression problem:  given a dataset of input-output pairs <img src="https://latex.codecogs.com/svg.image?{\color{Purple}&space;\mathbf{{(x_1,y_1),...,(x_n,y_n)}}}" title="https://latex.codecogs.com/svg.image?{\color{Purple} \mathbf{{(x_1,y_1),...,(x_n,y_n)}}}" align="center" />, how to learn <img src="https://latex.codecogs.com/svg.image?{\color{Purple}&space;\mathit{f}}" title="https://latex.codecogs.com/svg.image?{\color{Purple} \mathit{f}}" align="center" /> to predict the output <img src="https://latex.codecogs.com/svg.image?{\color{Purple}&space;\mathbf{\hat{y}&space;=&space;}f\mathbf{(x)}}" title="https://latex.codecogs.com/svg.image?{\color{Purple} \mathbf{\hat{y} = }f\mathbf{(x)}}" align="center"/> for any new input <img src="https://latex.codecogs.com/svg.image?{\color{Purple}&space;\mathbf{x}}" title="https://latex.codecogs.com/svg.image?{\color{Purple} \mathbf{x}}" align="center" />?

Here we generate an artificial dataset according to the following nonlinear function with the noise term : <img src="https://latex.codecogs.com/svg.image?\large&space;{\color{Purple}&space;\mathbf{\epsilon&space;}}" title="https://latex.codecogs.com/svg.image?\large {\color{Purple} \mathbf{\epsilon }}" align="center" />.

<img src="https://latex.codecogs.com/svg.image?\large&space;{\color{Purple}&space;\mathbf{y_i=2&space;\sin(x_i)&plus;x_{i}^{0.8}&plus;\epsilon&space;}}" title="https://latex.codecogs.com/svg.image?\large {\color{Purple} \mathbf{y_i=2 \sin(x_i)+x_{i}^{0.8}+\epsilon }}" align="center" />

where <img src="https://latex.codecogs.com/svg.image?\large&space;{\color{Purple}&space;\mathbf{\epsilon&space;}}" title="https://latex.codecogs.com/svg.image?\large {\color{Purple} \mathbf{\epsilon }}" align="center" /> obeys a normal distribution with zero mean and standard deviation 0.5. Both 50 training examples and 50 testing examples are generated. To better visualize the pattern of attention later, the training inputs are sorted.

### Average Pooling
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
We begin with perhaps the world’s “dumbest” estimator for this regression problem: using average pooling to average over all the training outputs:
<img src="https://latex.codecogs.com/svg.image?\large&space;{\color{Purple}f&space;\mathbf{(x)=&space;\frac{1}{n}\sum_{i=1}^{n}y_i&space;}}" title="https://latex.codecogs.com/svg.image?\large {\color{Purple}f \mathbf{(x)= \frac{1}{n}\sum_{i=1}^{n}y_i }}" align="center" />
 
which is plotted below. As we can see, this estimator is indeed not so smart.

### Nonparametric Attention Pooling
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

<img src="https://latex.codecogs.com/svg.image?\large&space;\large&space;{\color{Purple}f&space;\mathbf{(x)=&space;\sum_{i=1}^{n}\frac{K(x-x_i)}{\sum_{j=1}^{n}K(x-x_j)}y_i&space;}}" title="https://latex.codecogs.com/svg.image?\large \large {\color{Purple}f \mathbf{(x)= \sum_{i=1}^{n}\frac{K(x-x_i)}{\sum_{j=1}^{n}K(x-x_j)}y_i }}" />

where  **_K_** is a kernel. The estimator in (10.2.3) is called Nadaraya-Watson kernel regression. Here we will not dive into details of kernels. Recall the framework of attention mechanisms in Fig. 10.1.3. From the perspective of attention, we can rewrite (10.2.3) in a more generalized form of attention pooling: 

<img src="https://latex.codecogs.com/svg.image?\large&space;{\color{Purple}&space;f\mathbf{(x)=\sum^{n}_{i=0}\alpha(x,x_i)y_i}}" title="https://latex.codecogs.com/svg.image?\large {\color{Purple} f\mathbf{(x)=\sum^{n}_{i=0}\alpha(x,x_i)y_i}}" />


where **_x_** is the query and <img src="https://latex.codecogs.com/svg.image?{\color{Purple}&space;\mathbf{(x_i,y_i)}}" title="https://latex.codecogs.com/svg.image?{\color{Purple} \mathbf{(x_i,y_i)}}" align="center"/> is the key-value pair. Comparing (10.2.4) and (10.2.2), the attention pooling here is a weighted average of values **y<sub>i</sub>**. The attention weight <img src="https://latex.codecogs.com/svg.image?{\color{Purple}&space;\mathbf{\alpha(x,x_i)}" title="https://latex.codecogs.com/svg.image?{\color{Purple} \mathbf{\alpha(x,x_i)}" align="center" /> in (10.2.4) is assigned to the corresponding value **y<sub>i</sub>** based on the interaction between the query  and the key **x<sub>i</sub>** modeled by **&alpha;** . For any query, its attention weights over all the key-value pairs are a valid probability distribution: they are non-negative and sum up to one.

To gain intuitions of attention pooling, just consider a Gaussian kernel defined as

<img src="https://latex.codecogs.com/svg.image?\large&space;{\color{Purple}&space;K\mathbf{(u)=\frac{1}{\sqrt{2\pi}}exp&space;\left&space;(&space;-\frac{u^2}{2}&space;\right&space;)&space;}&space;&space;}" title="https://latex.codecogs.com/svg.image?\large {\color{Purple} K\mathbf{(u)=\frac{1}{\sqrt{2\pi}}exp \left ( -\frac{u^2}{2} \right ) } }" />

Plugging the Gaussian kernel into (10.2.4) and (10.2.3) gives

<img src="https://latex.codecogs.com/svg.image?\large&space;\\{\color{Purple}&space;f\mathbf{(x)=\sum^{n}_{i=1}\alpha(x,x_i)y_i}}&space;\\{\color{Purple}&space;f\mathbf{(x)=\sum^{n}_{i=1}\frac{exp\left&space;(&space;-\frac{1}{2}(x-x_i)^2&space;\right&space;)}{\sum^{n}_{j=1}exp\left&space;(&space;-\frac{1}{2}(x-x_i)^2&space;\right&space;)}y_i}}" title="https://latex.codecogs.com/svg.image?\large \\{\color{Purple} f\mathbf{(x)=\sum^{n}_{i=1}\alpha(x,x_i)y_i}} \\{\color{Purple} f\mathbf{(x)=\sum^{n}_{i=1}\frac{exp\left ( -\frac{1}{2}(x-x_i)^2 \right )}{\sum^{n}_{j=1}exp\left ( -\frac{1}{2}(x-x_i)^2 \right )}y_i}}" />

<img src="https://latex.codecogs.com/svg.image?\large&space;\\{\color{Purple}&space;f\mathbf{(x)=\sum^{n}_{i=1}softmax\left&space;(-\frac{1}{2}(x-x_i)^2&space;\right&space;)y_i.}}" title="https://latex.codecogs.com/svg.image?\large \\{\color{Purple} f\mathbf{(x)=\sum^{n}_{i=1}softmax\left (-\frac{1}{2}(x-x_i)^2 \right )y_i.}}" />

### Parametric Attention Pooling
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

<img src="https://latex.codecogs.com/svg.image?\large&space;\\{\color{Purple}&space;f\mathbf{(x)=\sum^{n}_{i=1}\alpha(x,x_i)y_i,}}&space;\\{\color{Purple}&space;f\mathbf{(x)=\sum^{n}_{i=1}\frac{exp\left&space;(&space;-\frac{1}{2}\left&space;((x-x_i)w&space;\right&space;)^2\right&space;)}{\sum^{n}_{j=1}exp\left&space;(&space;-\frac{1}{2}\left&space;((x-x_i)w&space;\right&space;)^2\right&space;)}y_i,}}\\{\color{Purple}&space;f\mathbf{(x)=\sum^{n}_{i=1}softmax&space;\left&space;(&space;-\frac{1}{2}\left&space;((x-x_i)w&space;\right&space;)^2\right&space;)y_i}}&space;" title="https://latex.codecogs.com/svg.image?\large \\{\color{Purple} f\mathbf{(x)=\sum^{n}_{i=1}\alpha(x,x_i)y_i,}} \\{\color{Purple} f\mathbf{(x)=\sum^{n}_{i=1}\frac{exp\left ( -\frac{1}{2}\left ((x-x_i)w \right )^2\right )}{\sum^{n}_{j=1}exp\left ( -\frac{1}{2}\left ((x-x_i)w \right )^2\right )}y_i,}}\\{\color{Purple} f\mathbf{(x)=\sum^{n}_{i=1}softmax \left ( -\frac{1}{2}\left ((x-x_i)w \right )^2\right )y_i}} " />


 ### Batch Matrix Multiplication
 ![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
To more efficiently compute attention for minibatches, we can leverage batch matrix multiplication utilities provided by deep learning frameworks.

Suppose that the first minibatch contains **n** matrices **X<sub>1</sub>,..., X<sub>n</sub>** of shape **a &times; b** , and the second minibatch contains **n** matrices **Y<sub>1</sub>,..., Y<sub>n</sub>** of shape **b &times; c**. Their batch matrix multiplication results in **n** matrices **X<sub>1</sub>Y<sub>1</sub>,..., X<sub>n</sub>Y<sub>n</sub>** of shape **a &times; c**. Therefore, given two tensors of shape **( n, a, b )** and **( n, b, c )**, the shape of their batch matrix multiplication output is **( n, a, c )**.

## Summary
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* **Nadaraya-Watson** _kernel regression_ is an example of **_machine learning with attention mechanisms_**.
* The attention pooling of **Nadaraya-Watson** kernel regression is a **weighted average** of the training outputs. From the attention perspective, the attention weight is assigned to a value based on a function of a query and the key that is paired with the value.
* Attention pooling can be either nonparametric or parametric.


## Attention Scoring Functions
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
In the above we used a **Gaussian kernel** to model interactions between **queries** and **keys**. Treating the **exponent** of the Gaussian kernel in (10.2.6) as an attention scoring function (or scoring function for short), the results of this function were essentially fed into a softmax operation. As a result, we obtained a probability distribution (attention weights) over values that are paired with keys. In the end, the output of the attention pooling is simply a weighted sum of the values based on these attention weights.

At a high level, we can use the above algorithm to instantiate the framework of attention mechanisms in Fig. 10.1.3. Denoting an attention scoring function by , Fig. 10.3.1 illustrates how the output of attention pooling can be computed as a weighted sum of values. Since attention weights are a probability distribution, the weighted sum is essentially a weighted average.

<img src="https://user-images.githubusercontent.com/12748752/170890210-d8b92410-afa4-493d-8500-a9e00a78b5f4.png" width=90% />

<ins><i><b>Computing the output of attention pooling as a weighted average of values.</b></i></ins>





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

## Todo
* [Dive into DL](https://d2l.ai/chapter_attention-mechanisms/nadaraya-watson.html)

## References
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)



