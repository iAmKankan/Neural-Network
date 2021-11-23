## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

## The Perceptron
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* The Perceptron is one of the simplest ANN architectures, invented in 1957 by Frank Rosenblatt. 
* It is based on a slightly different artificial neuron called a **Threshold Logic Unit (TLU)**, or sometimes a **Linear Threshold Unit (LTU)**: the inputs and output are now numbers (instead of binary on/off values) and each input connection is associated with a weight.
* A single TLU can be used for simple linear binary classification. 
> * It computes a linear combination of the inputs and if the result exceeds a threshold, it outputs the positive class or else outputs the negative class (just like a Logistic Regression classifier or a linear SVM).

### Training Perceptron
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* The Perceptron training algorithm proposed by Frank Rosenblatt was inspired by Hebb’s rule.
* **Donald Hebb rule** suggested that when a biological neuron often triggers another neuron, the connection between these two neurons grows stronger.

> #### <img src="http://latex.codecogs.com/svg.image?w_{i,j}^{(Next\&space;Step)}&space;=&space;w_{i,j}&space;&plus;&space;\eta(y_j&space;-&space;\hat{y_j})x_i" title="w_{i,j}^{(Next\ Step)} = w_{i,j} + \eta(y_j - \hat{y_j})x_i" width=45% />

> #### Where
> 
>> <img src="https://latex.codecogs.com/svg.image?&space;w_{i,j}&space;\textrm{&space;:&space;connection&space;weight&space;between}&space;\&space;\&space;i^{th}&space;&space;\textrm{input&space;neuron&space;and&space;}&space;j^{th}&space;&space;\textrm{&space;output&space;neuron}" title=" w_{i,j} \textrm{ : connection weight between} \ \ i^{th} \textrm{input neuron and } j^{th} \textrm{ output neuron}" />.  
>>
>> <img src="https://latex.codecogs.com/svg.image?x_i&space;:&space;i^{th}\textrm{&space;input&space;value}" title="x_i : i^{th}\textrm{ input value}" />.
>>
>> <img src="https://latex.codecogs.com/svg.image?\hat{y_j}&space;:&space;\textrm{output&space;of}&space;\&space;j^{th}\&space;\textrm{&space;output&space;}" title="\hat{y_j} : \textrm{output of} \ j^{th}\ \textrm{ output }" />.
>>
>> <img src="https://latex.codecogs.com/svg.image?y_j&space;:&space;\textrm{target&space;output&space;of}\&space;\&space;j^{th}&space;\textrm{&space;output&space;neuron}" title="y_j : \textrm{target output of}\ \ j^{th} \textrm{ output neuron}" />.
>>
>> <img src="https://latex.codecogs.com/svg.image?\eta&space;:&space;\textrm{learning&space;rate}" title="\eta : \textrm{learning rate}" />.  

> #### It can also be written as for jth element of w vector 
> <img src="https://latex.codecogs.com/svg.image?w_j&space;=&space;w_j&space;&plus;&space;\triangle&space;w_j" title="w_j = w_j + \triangle w_j" />.
>
> <img src="https://latex.codecogs.com/svg.image?where,\&space;\triangle&space;w_j&space;=&space;&space;\eta(y^{(i)}&space;-&space;\hat{y_j}^{(i)})x_j^{(i)}" title="where,\ \triangle w_j = \eta(y^{(i)} - \hat{y_j}^{(i)})x_j^{(i)}" />.




* Scikit-Learn provides a Perceptron class that implements a single TLU network. 
* It can be used pretty much as you would expect—for example, on the iris dataset 
```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
iris = load_iris()
X = iris.data[:, (2, 3)] # petal length, petal width
y = (iris.target == 0).astype(np.int) # Iris Setosa?
per_clf = Perceptron()
per_clf.fit(X, y)
y_pred = per_clf.predict([[2, 0.5]])
```







## Multi-Layer Perceptron
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* An MLP is composed of -
   * **One (passthrough) input layer,**
   * **One or more layers of TLUs- called hidden layers,**
   * **One final layer of TLUs called the output layer.**

* The layers close to the input layer are usually called the **lower layers**, and the ones close to the outputs are usually called the **upper layers**. 
* Every layer except the output layer includes **a bias neuron** and is fully connected to the next layer.

![](https://user-images.githubusercontent.com/12748752/143045465-2fe26cb7-48ea-4590-b381-24215f014004.png)

<img src="https://user-images.githubusercontent.com/12748752/143045465-2fe26cb7-48ea-4590-b381-24215f014004.png" />


