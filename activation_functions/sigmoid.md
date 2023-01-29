## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)


## Sigmoid
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
Let see **Sigmoid non-linear function** -


<p align="center">
 <img src="https://user-images.githubusercontent.com/12748752/215315005-24d27488-ece0-4512-8046-fc0e8aa9427b.png"/>
 <br>
 <ins> <i>Horizontal axis is - $\Large{\color{Purple}x}$ axis ;  virtical axis is - $\Large{\color{Purple}\sigma(x)}$ axis </i></ins>
</p>

So, we will look at the sigmoid non-linear activation now, here is the ana- lytical expression for this sigmoid non-linear activation, okay and this is the plot of the function, you can see that, so this horizontal axis is - $\Large{\color{Purple}x}$ axis and  virtical axis is - $\Large{\color{Purple}\sigma(x)}$ axis.

So, we can see that for **large positive value of x** the sigmoid function tends to **1**, and for **large negative values of x** it tends to **0**. 

Thing to notice here, it becomes **flat** **for large positive values of x** as well as for **large negative values of x**. 

We use **x** and **z** interchangeably with this function $\Large{\color{Purple} \mathbf{\sigma(z)} = \mathbf{\frac{1}{1+e^{-z}}}}$ . 

Here **z** is the input to the **sigmoid function** is the **linear combination** of your **input features** with the **weights**. 
  * It says that if $\Large{\color{Purple}z}$ is very large means $\Large{\color{Purple}\Sigma W_i x_i }$ is very large then **either magnitude** of that get **very large positive** as well as **very large negative** numbers leads the sigmoid function being **saturated**.
  * Which means that the **gradient**,  in this case $\Large{\color{Purple}\frac{\partial z(\sigma)}{\partial z} \sim 0}$   or very close to 0 or very small number.
  * That scenario leads to negligible or 0 updates to your weights in [Backpropagation ↗️](https://github.com/iAmKankan/Neural-Network/blob/main/backpropagation/README.md) ([Vanishing Gradient problem ↗️](https://github.com/iAmKankan/Neural-Network/tree/main/activation_functions#vanishing-gradients-the-rnn-version-%EF%B8%8F))






$$\Large{\color{Purple} \mathbf{z = \Sigma W_i x_i} }\ \ \ \normalsize  \[{\textit{ Here we are not considering the bias term  }} W_0 \]$$ 

* A sigmoid function is a mathematical function having a characteristic "**_S_**"-shaped curve or **sigmoid curve**.
* A common example of a sigmoid function is the **logistic function**.
* The term "**_Sigmoid function_**" is used as an _alias_ for the **_logistic function_**.
 
### The formula: 

$$\Huge{\color{Purple} \mathbf{\sigma(x)} = \mathbf{\frac{1}{1+e^{-x}}}}$$


### Properties: 
<img src="https://latex.codecogs.com/svg.image?\large&space;\begin{matrix}{\color{Blue}&space;\textbf{Domain(input&space;range)}\&space;\&space;\sigma(x)}&space;&&space;{\color{Blue}&space;\textbf{Range(output&space;range)}&space;\&space;\&space;\sigma(x)}&&space;{\color{Blue}&space;\textbf{Thresold&space;value}&space;\&space;\&space;\sigma(x)}&space;&&space;{\color{Blue}&space;\textbf{Derivative}&space;\&space;\&space;\frac{\partial&space;}{\partial&space;x}&space;\sigma(x)}&space;\\&space;{\color{DarkRed}&space;(-\infty,\infty)}&space;&&space;{\color{DarkRed}\textbf{(0,1)}&space;}&{\color{DarkRed}&space;\textbf{0.5}}&{\color{DarkRed}&space;\textbf{0.25}}&space;\\&space;\end{matrix}" title="https://latex.codecogs.com/svg.image?\large \begin{matrix}{\color{Blue} \textbf{Domain(input range)}\ \ \sigma(x)} & {\color{Blue} \textbf{Range(output range)} \ \ \sigma(x)}& {\color{Blue} \textbf{Thresold value} \ \ \sigma(x)} & {\color{Blue} \textbf{Derivative} \ \ \frac{\partial }{\partial x} \sigma(x)} \\ {\color{DarkRed} (-\infty,\infty)} & {\color{DarkRed}\textbf{(0,1)} }&{\color{DarkRed} \textbf{0.5}}&{\color{DarkRed} \textbf{0.25}} \\ \end{matrix}" />

### The graph: 
<img src="https://user-images.githubusercontent.com/12748752/167588897-79a754f8-e4db-48c8-babf-3203d5d3e9bc.png" width=60% align="center"/>

The Sigmoid function is the most frequently used activation function in the beginning of deep learning. It is a smoothing function that is easy to derive.

In the sigmoid function, we can see that its output is in the open interval (0,1). We can think of probability, but in the strict sense, don't treat it as probability. The sigmoid function was once more popular. It can be thought of as the firing rate of a neuron. In the middle where the slope is relatively large, it is the sensitive area of the neuron. On the sides where the slope is very gentle, it is the neuron's inhibitory area.

The function itself has certain defects.

1) When the input is slightly away from the coordinate origin, the gradient of the function becomes very small, almost zero. In the process of neural network backpropagation, we all use the chain rule of differential to calculate the differential of each weight w. When the backpropagation passes through the sigmod function, the differential on this chain is very small. Moreover, it may pass through many sigmod functions, which will eventually cause the weight **_W_** to have little effect on the loss function, which is not conducive to the optimization of the weight. This The problem is called gradient saturation or gradient dispersion.

2) The function output is not centered on 0, which will reduce the efficiency of weight update.

3) The sigmod function performs exponential operations, which is slower for computers.


#### Advantages of Sigmoid Function : -

1. Smooth gradient, preventing “jumps” in output values.
2. Output values bound between 0 and 1, normalizing the output of each neuron.
3. Clear predictions, i.e very close to 1 or 0.

#### Sigmoid has three major disadvantages:
* Prone to gradient vanishing
* Function output is not zero-centered
* Power operations are relatively time consuming

#### The Sigmoid Curve
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

 <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/480px-Logistic-curve.svg.png"/>
 
### Derivative of Sigmoid Activation function 
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
