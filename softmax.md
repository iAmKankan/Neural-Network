## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

## Introduction
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
> ### Softmax regression (or multinomial logistic regression) is a generalization of logistic regression to the case where we want to handle multiple classes.
* In logistic regression we assumed that the labels were binary: <img src="https://render.githubusercontent.com/render/math?math=y^{(i)} \in\ \{0,1\}">.
* Softmax regression allows us to handle <img src="https://render.githubusercontent.com/render/math?math=y^{(i)} \in\ \{1,..,K\}"> where _**K**_ is the number of classes.


* For binary logistic regression we had <img src="http://www.sciweavers.org/tex2img.php?eq=f%28x%3Bw%29%3D%5Csigma%28w%5E%5Ctop%20x%29%3D%5Cfrac%7B1%7D%7B1%2Be%5E%7B-w%5E%5Ctop%20x%7D%7D%5C%20%5C%20with%20%5C%20y%5E%7B%28i%29%7D%20%5Cin%5C%20%5C%7B0%2C1%5C%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="f(x;w)=\sigma(w^\top x)=\frac{1}{1+e^{-w^\top x}}\ \ with \ y^{(i)} \in\ \{0,1\}" width="408" height="44" />

* We interpreted the output as _**P(y = 1|x; w), implying P(y=0|x; w) = 1-f(x; w)**_ (if we want class 0 we just need to minus 1 from the class 1) .

* For the multiclass setting we now have <img src="https://render.githubusercontent.com/render/math?math=y^{(i)} \in\ \{1,..,K\}">.

* **Idea:** Instead of just outputting a single value for the positive class, let's output a vector of probabilities for each class:

<img src="http://www.sciweavers.org/tex2img.php?eq=f%28x%3B%20W%29%3D%20%0A%5Cbegin%7Bbmatrix%7D%0A%20P%28y%20%3D%201%7Cx%3BW_1%29%5C%5C%0AP%28y%20%3D%202%7Cx%3BW_2%29%20%5C%5C%0A...%20%5C%5C%0AP%28y%20%3D%20K%7Cx%3BW_K%29%20%5C%5C%0A%5Cend%7Bbmatrix%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="f(x; W)= \begin{bmatrix} P(y = 1|x;W_1)\\P(y = 2|x;W_2) \\... \\P(y = K|x;W_K) \\\end{bmatrix}" width="229" height="79" />


* We will now build up to a model that does this.

* Each element in _**f(x;W)**_ should be a "score" for how well input **x** matches that class

* For input **x** let's set the store for class _**k**_ to <img src="http://www.sciweavers.org/tex2img.php?eq=w_%7Bk%7D%5E%7B%5Ctop%7Dx&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="w_{k}^{\top}x" width="42" height="25" />



* But probabilities need to be positive. So let's take the exponential: <img src="http://www.sciweavers.org/tex2img.php?eq=e%5E%7Bw_%7Bk%7D%5E%7B%5Ctop%7Dx%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="e^{w_{k}^{\top}x}" width="43" height="22" />




This gives us the softmax regression model



## References
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* [Deeplearning Stanford University](http://deeplearning.stanford.edu/tutorial/supervised/SoftmaxRegression/)
* [Herman Kamper](https://www.kamperh.com/data414/)
