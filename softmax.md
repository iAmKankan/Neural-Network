## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

## Introduction
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
> ### Softmax regression (or multinomial logistic regression) is a generalization of logistic regression to the case where we want to handle multiple classes.
* In logistic regression we assumed that the labels were binary: <img src="https://render.githubusercontent.com/render/math?math=y^{(i)} \in\ \{0,1\}">.
* Softmax regression allows us to handle <img src="https://render.githubusercontent.com/render/math?math=y^{(i)} \in\ \{1,..,K\}"> where _**K**_ is the number of classes.

<img src="https://render.githubusercontent.com/render/math?math=f\mathrm{(x;w)\ =\ \ \sigma(w^{\top}x)\ = \frac{1}{1+e^{-w^\top x}}\ \ with\ \ y\in\ \{0,1\}}">.

* For binary logistic regression we had 
---
<img src="http://www.sciweavers.org/tex2img.php?eq=%5Csigma%20%5Cmathrm%7B%28w%5E%7B%5Ctop%7Dx%29%5C%20%3D%20%5Cfrac%7B1%7D%7B1%2Be%5E%7B-w%5E%5Ctop%20x%7D%7D%5C%20%5C%20with%5C%20%5C%20y%5Cin%5C%20%5C%7B0%2C1%5C%7D%7D&bc=Transparent&fc=Black&im=png&fs=12&ff=cmbright&edit=0" align="center" border="0" alt="\sigma \mathrm{(w^{\top}x)\ = \frac{1}{1+e^{-w^\top x}}\ \ with\ \ y\in\ \{0,1\}}" width="75" height="18" />

• We interpreted the output as P(y = 1|x; w), implying P(y=0|x; w) = 1-f(x; w).

• For the multiclass setting we now have y E {1,2,..., K}.

• Idea: Instead of just outputting a single value for the positive class, let's output a vector of probabilities for each class:

P(y = 1|x;W)

f(x; W)

P(y = 2x;W)

P(y = Kx; W

. We will now build up to a model that does this.

Softmax regression

31°C Haze 4 ENG 03:33 PM

E




## References
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* [Deeplearning Stanford University](http://deeplearning.stanford.edu/tutorial/supervised/SoftmaxRegression/)
* [Herman Kamper](https://www.kamperh.com/data414/)
