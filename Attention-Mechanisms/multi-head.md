## Index:
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

## Multi-Head Attention
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

* In practice, given the same set of **queries**, **keys**, and **values** we may want our model to combine knowledge from different behaviors of the same attention mechanism, such as capturing dependencies of various ranges (e.g., **shorter-range** vs. **longer-range**) within a sequence. 
* Thus, it may be beneficial to allow our attention mechanism _to jointly use different representation subspaces_ of **queries, keys, and values**.

To this end, instead of performing a single attention **pooling**, **queries**, **keys** and **values** can be transformed with  independently learned linear projections. Then these **_h_** projected queries, keys, and values are fed into **attention pooling in parallel**. 

In the end, **_h_** attention pooling outputs are concatenated and transformed with another learned linear projection to produce the final output. 

This design is called **_multi-head attention_**, where each **_h_** of the  attention pooling outputs is a head. Using **fully-connected layers** to perform **learnable linear transformations**.

<img src="https://user-images.githubusercontent.com/12748752/170055315-b69b2b13-f3a5-44c6-8a6a-6a4655359f80.png" width=60%/>
<p align="center"><ins><i><b>Multi-head attention, where multiple heads are concatenated then linearly transformed.</b></i></ins></p>

### Model
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
Let us formalize this model mathematically. 

Given a **query** <img src="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{q&space;\in&space;\mathbb{R}^{d_q}}&space;}" title="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{q \in \mathbb{R}^{d_q}} }" align="center"/> , **a key** <img src="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{k&space;\in&space;\mathbb{R}^{d_k}}&space;}" title="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{k \in \mathbb{R}^{d_k}} }" /> , and **a value**  <img src="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{v&space;\in&space;\mathbb{R}^{d_v}}&space;}" title="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{v \in \mathbb{R}^{d_v}} }" align="center" /> , each attention head <img src="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{h_i(i=1,\dots,&space;h)}&space;}" title="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{h_i(i=1,\dots, h)} }" align="center"/> is computed as


<img src="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{h_i=}\mathit{f}\mathbf{\left&space;(&space;W^{(q)}_{i}q,W^{(k)}_{i}k,W^{(v)}_{i}v&space;\right&space;)\in&space;\mathbb{R}^{p_v},}&space;}" title="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{h_i=}\mathit{f}\mathbf{\left ( W^{(q)}_{i}q,W^{(k)}_{i}k,W^{(v)}_{i}v \right )\in \mathbb{R}^{p_v},} }" />

where learnable parameters <img src="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf&space;{W^{(q)}_{i}&space;\in&space;\mathbb{R}^{p_q&space;\times&space;d_q},W^{(k)}_{i}&space;\in&space;\mathbb{R}^{p_k&space;\times&space;d_k}{\color{Black}\mathrm{\&space;and\&space;}&space;}W^{(v)}_{i}&space;\in&space;\mathbb{R}^{p_v&space;\times&space;d_v}&space;{\color{Black}\mathrm{\&space;and\&space;}&space;}\mathit{f}&space;}&space;}" title="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf {W^{(q)}_{i} \in \mathbb{R}^{p_q \times d_q},W^{(k)}_{i} \in \mathbb{R}^{p_k \times d_k}{\color{Black}\mathrm{\ and\ } }W^{(v)}_{i} \in \mathbb{R}^{p_v \times d_v} {\color{Black}\mathrm{\ and\ } }\mathit{f} } }" align="center"/> is attention pooling, such as additive attention and scaled dot-product attention in Section 10.3. The multi-head attention output is another linear transformation via learnable parameters <img src="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf&space;{W_{o}&space;\in&space;\mathbb{R}^{p_o&space;\times&space;hp_{v}}&space;}&space;}" title="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf {W_{o} \in \mathbb{R}^{p_o \times hp_{v}} } }" align="center"/> of the concatenation of <img src="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf&space;{h}}&space;" title="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf {h}} " align="center" /> heads:
 
<img src="https://latex.codecogs.com/svg.image?\large&space;{\color{Purple}\mathbf&space;{W_{o}}&space;&space;\begin{bmatrix}&space;\mathbf{h_1}\\&space;\vdots&space;\\&space;\mathbf{h_h}\end{bmatrix}\mathbf&space;{\in&space;\mathbb{R}^{p_o}&space;}&space;}&space;" title="https://latex.codecogs.com/svg.image?\large {\color{Purple}\mathbf {W_{o}} \begin{bmatrix} \mathbf{h_1}\\ \vdots \\ \mathbf{h_h}\end{bmatrix}\mathbf {\in \mathbb{R}^{p_o} } } " />
 
Based on this design, each head may attend to different parts of the input. More sophisticated functions than the simple weighted average can be expressed.
## Summary:
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* **_Multi-head attention_** combines knowledge of the same **attention pooling** via different representation subspaces of **queries**, **keys**, and **values**.
* To compute multiple heads of multi-head attention in parallel, proper tensor manipulation is needed. 
