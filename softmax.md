## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

## Introduction
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
> ### Softmax regression (or multinomial logistic regression) is a generalization of logistic regression to the case where we want to handle multiple classes.
* In logistic regression we assumed that the labels were binary: <img src="https://render.githubusercontent.com/render/math?math=y^{(i)} \in\ \{0,1\}">.
* Softmax regression allows us to handle <img src="https://render.githubusercontent.com/render/math?math=y^{(i)} \in\ \{1,..,K\}"> where _**K**_ is the number of classes.


* For binary logistic regression we had <img src="http://www.sciweavers.org/upload/Tex2Img_1637585715/render.png" width="408" height="44" />

* We interpreted the output as _**P(y = 1|x; w), implying P(y=0|x; w) = 1-f(x; w)**_ (if we want class 0 we just need to minus 1 from the class 1) .
![687474703a2f2f7777772e736369776561766572732e6f72672f75706c6f61642f54657832496d675f313633373538353731352f72656e6465722e706e67](https://user-images.githubusercontent.com/12748752/142930556-4740bde9-3f91-4692-90ee-c063f9f4576f.png)


![687474703a2f2f7777772e736369776561766572732e6f72672f75706c6f61642f54657832496d675f313633373538353731352f72656e6465722e706e67](https://user-images.githubusercontent.com/12748752/142930565-7a511466-2d3e-4d6e-b8c5-831f11389177.png)


* For the multiclass setting we now have <img src="https://render.githubusercontent.com/render/math?math=y^{(i)} \in\ \{1,..,K\}">.

* **Idea:** Instead of just outputting a single value for the positive class, let's output a vector of probabilities for each class:

<img src="http://latex.codecogs.com/svg.image?f(x;&space;W)=&space;\begin{bmatrix}&space;P(y&space;=&space;1|x;W_1)\\P(y&space;=&space;2|x;W_2)&space;\\...&space;\\P(y&space;=&space;K|x;W_K)&space;\\\end{bmatrix}" title="f(x; W)= \begin{bmatrix} P(y = 1|x;W_1)\\P(y = 2|x;W_2) \\... \\P(y = K|x;W_K) \\\end{bmatrix}" />



* We will now build up to a model that does this.

* Each element in _**f(x;W)**_ should be a "score" for how well input **x** matches that class

* For input **x** let's set the store for class _**k**_ to <img src="http://latex.codecogs.com/svg.image?{w^\top_k&space;x}" title="{w^\top_k x}" />



* But probabilities need to be positive. So let's take the exponential: <img src="http://latex.codecogs.com/svg.image?e^{w^\top_k&space;x}" title="e^{w^\top_k x}" />


* But probabilities need to sum to one. So let's normalise 
<img src="http://latex.codecogs.com/svg.image?P(\mathrm{y=k\&space;|\&space;x;W})=&space;\frac{e^{w^\top_k&space;x}}{\sum_{j=1}^{k}e^{w^\top_j&space;x}}" title="P(\mathrm{y=k\ |\ x;W})= \frac{e^{w^\top_k x}}{\sum_{j=1}^{k}e^{w^\top_j x}}" />

This gives us the softmax regression model



## References
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* [Deeplearning Stanford University](http://deeplearning.stanford.edu/tutorial/supervised/SoftmaxRegression/)
* [Herman Kamper](https://www.kamperh.com/data414/)
