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

* For the multiclass setting we now have <img src="https://render.githubusercontent.com/render/math?math=y^{(i)} \in\ \{1,..,K\}">.

* **Idea:** Instead of just outputting a single value for the positive class, let's output a vector of probabilities for each class:

<img src="http://www.sciweavers.org/upload/Tex2Img_1637585596/render.png" width="229" height="79" />


* We will now build up to a model that does this.

* Each element in _**f(x;W)**_ should be a "score" for how well input **x** matches that class

* For input **x** let's set the store for class _**k**_ to <img src="http://www.sciweavers.org/upload/Tex2Img_1637585447/render.png" width="42" height="25" />



* But probabilities need to be positive. So let's take the exponential: <img src="http://www.sciweavers.org/upload/Tex2Img_1637585389/render.png" width="43" height="22" />


* But probabilities need to sum to one. So let's normalise 
<img src="http://www.sciweavers.org/upload/Tex2Img_1637584945/render.png" width="218" height="67" />


This gives us the softmax regression model



## References
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* [Deeplearning Stanford University](http://deeplearning.stanford.edu/tutorial/supervised/SoftmaxRegression/)
* [Herman Kamper](https://www.kamperh.com/data414/)
