## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

### Sigmoid
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* A sigmoid function is a mathematical function having a characteristic "S"-shaped curve or sigmoid curve.
* A common example of a sigmoid function is the logistic function.
* The term "sigmoid function" is used as an alias for the logistic function.


#### The Sigmoid Curve
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

 <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/480px-Logistic-curve.svg.png"/>
 
#### The Sigmoid Formula
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

<img src="https://latex.codecogs.com/svg.image?\mathit{S(x)}\&space;=&space;\&space;\frac{1}{1&plus;e^{-x}}" title="\mathit{S(x)}\ = \ \frac{1}{1+e^{-x}}" width=20% />

#### Properties of Sigmoid function
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* As it produce value with rage between **0** and **1** it is computationally less intensive.
* Converging the solution faster.
<img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}\textbf{Domain&space;for(input&space;range)}\&space;\&space;\sigma(x)&space;&&space;\textbf{Range&space;for(output&space;range)}&space;\&space;\&space;\sigma(x)&space;&space;\\&space;(-\infty,\infty)&space;&(0,1)&space;&space;\\&space;\end{bmatrix}&space;" title="\begin{bmatrix}\textbf{Domain for(input range)}\ \ \sigma(x) & \textbf{Range for(output range)} \ \ \sigma(x) \\ (-\infty,\infty) &(0,1) \\ \end{bmatrix} " />

#### Derivative of Sigmoid Activation function 
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* Inorder to do so we need to introduce **Quotient rule formula in Differentiation**

> #### _**Quotient rule**_
>> * <img src="https://latex.codecogs.com/svg.image?\frac{d}{dx}f=\frac{(denominator&space;*&space;\frac{d}{dx}numerator)&space;-&space;(numerator&space;*&space;\frac{d}{dx}denominator)}{denominator^2}" title="\frac{d}{dx}f=\frac{(denominator * \frac{d}{dx}numerator) - (numerator * \frac{d}{dx}denominator)}{denominator^2}" />

> #### _**Derivatives of Sigmoid Function**_
* <img src="https://latex.codecogs.com/svg.image?\frac{d}{dx}S(x)=\frac{1}{1&plus;e^{{-x}}}" title="\frac{d}{dx}S(x)=\frac{1}{1+e^{{-x}}}" />
* <img src="https://latex.codecogs.com/svg.image?\frac{d}{dx}S(x)=\frac{((1&plus;e^{-x})&space;*&space;\frac{d}{dx}1)&space;-&space;(1&space;*&space;\frac{d}{dx}(1&plus;e^{-x}))}{({1&plus;e^{-x}})^2}" title="\frac{d}{dx}S(x)=\frac{((1+e^{-x}) * \frac{d}{dx}1) - (1 * \frac{d}{dx}(1+e^{-x}))}{({1+e^{-x}})^2}" />
* <img src="https://latex.codecogs.com/svg.image?\frac{d}{dx}S(x)=\frac{({1&plus;e^{-x}})&space;(0)&space;-&space;(1)&space;(-e^{-x})}{({1&plus;e^{-x}})^2}&space;" title="\frac{d}{dx}S(x)=\frac{({1+e^{-x}}) (0) - (1) (-e^{-x})}{({1+e^{-x}})^2} " />
* <img src="https://latex.codecogs.com/svg.image?\frac{d}{dx}S(x)=\frac{e^{-x}}{({1&plus;e^{-x}})^2}&space;" title="\frac{d}{dx}S(x)=\frac{e^{-x}}{({1+e^{-x}})^2} " />
* <img src="https://latex.codecogs.com/svg.image?\frac{d}{dx}S(x)=\frac{1-1&plus;e^{-x}}{({1&plus;e^{-x}})^2}&space;" title="\frac{d}{dx}S(x)=\frac{1-1+e^{-x}}{({1+e^{-x}})^2} " />
* <img src="https://latex.codecogs.com/svg.image?\frac{d}{dx}S(x)=\frac{1&plus;e^{-x}}{({1&plus;e^{-x}})^2}-\frac{1}{(1&plus;e^{-x})^2}" title="\frac{d}{dx}S(x)=\frac{1+e^{-x}}{({1+e^{-x}})^2}-\frac{1}{(1+e^{-x})^2}" />
* <img src="https://latex.codecogs.com/svg.image?\frac{d}{dx}S(x)=\frac{1}{({1&plus;e^{-x}})}-\frac{1}{(1&plus;e^{-x})^2}" title="\frac{d}{dx}S(x)=\frac{1}{({1+e^{-x}})}-\frac{1}{(1+e^{-x})^2}" />
* <img src="https://latex.codecogs.com/svg.image?\frac{d}{dx}S(x)=\frac{1}{({1&plus;e^{-x}})}\left&space;(&space;1-\frac{1}{(1&plus;e^{-x})}&space;\right&space;)" title="\frac{d}{dx}S(x)=\frac{1}{({1+e^{-x}})}\left ( 1-\frac{1}{(1+e^{-x})} \right )" />
