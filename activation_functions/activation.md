## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* [Activation Function](#activation-function)
   * [Sigmoid](#sigmoid)
   * [TanH](#tanh)






## Activation Function
![dark](https://user-images.githubusercontent.com/12748752/136802585-2ef5b7ff-ddbc-417f-b963-ca233db3ded1.png)

### Why do we need activation functions in the first place
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
* If you chain several linear transformations, all you get is a linear transformation.
> **For example: Say we have f(x) and g(x) then Then chaining these two linear functions gives you another linear function f(g(x)).**
>> f(x) = 2 x + 3 
>> 
>> g(x) = 5 x - 1 
>> 
>> f(g(x)) = 2(5 x - 1) + 3 = 10 x + 1.
>
> 
> 
* So, if you don’t have some non-linearity between layers, then even a deep stack of layers is equivalent to a single layer.
* You cannot solve very complex problems with that.

### Sigmoid
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
* A sigmoid function is a mathematical function having a characteristic "S"-shaped curve or sigmoid curve.
* A common example of a sigmoid function is the logistic function.
* The term "sigmoid function" is used as an alias for the logistic function.


#### The Sigmoid Curve
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)

 <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/480px-Logistic-curve.svg.png"/>
 
#### The Sigmoid Formula
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)

<img src="https://latex.codecogs.com/svg.image?\mathit{S(x)}\&space;=&space;\&space;\frac{1}{1&plus;e^{-x}}" title="\mathit{S(x)}\ = \ \frac{1}{1+e^{-x}}" width=20% />

#### Properties of Sigmoid function
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
* As it produce value with rage between **0** and **1** it is computationally less intensive.
* Converging the solution faster.
<img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}\textbf{Domain&space;for(input&space;range)}\&space;\&space;\sigma(x)&space;&&space;\textbf{Range&space;for(output&space;range)}&space;\&space;\&space;\sigma(x)&space;&space;\\&space;(-\infty,\infty)&space;&(0,1)&space;&space;\\&space;\end{bmatrix}&space;" title="\begin{bmatrix}\textbf{Domain for(input range)}\ \ \sigma(x) & \textbf{Range for(output range)} \ \ \sigma(x) \\ (-\infty,\infty) &(0,1) \\ \end{bmatrix} " />

#### Derivative of the Sigmoid Activation function 
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
* Inorder to do so we need to introduce **Quotient rule formula in Differentiation**
 > 


