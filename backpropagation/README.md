## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

## Backpropagation

In 1986, **David Rumelhart**, **Geoffrey Hinton** and **Ronald Williams** published a paper that introduced the **_backpropagation_** training algorithm. 

> Backpropagation is a technique for _computing the gradients automatically_ using _Gradient Descent_ in just two passes through the network (one **forward**, one **backward**), the **backpropagation algorithm** is able find out how each _connection weight_ and each _bias term_ should be **_tweaked_** in order to reduce the _error_. Once it has these gradients, it just performs a regular Gradient Descent step, and the whole process is repeated until the network converges to the solution.

<img src="https://latex.codecogs.com/svg.image?\large&space;{\color{DarkOrange}&space;\mathbf{or}}" title="https://latex.codecogs.com/svg.image?\large {\color{DarkOrange} \mathbf{or}}" align="center"/>

> Backpropagation refers to the method of calculating the gradient of neural network parameters. In short, the method traverses the network in _reverse order_, from the **output** to the **input layer**, according to the chain rule from calculus. The algorithm stores any intermediate variables (partial derivatives) required while calculating the gradient with respect to some parameters.
  
  <img src="https://user-images.githubusercontent.com/12748752/166812953-a0c9f542-1886-4f0c-bad4-a4cf815feeeb.png" width=70% align="center" />

### The Algorithm
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* It contains two parts - **forward** and **backward**
* It handles one **mini-batch** at a time (for example, containing 32 instances each), and it goes through the full training set multiple times. ( 1 pass = one epoch)
#### Forward pass:
##### Step 1:
* Each mini-batch is passed to the network’s **input layer**, which sends it to the **first hidden layer**. The algorithm then computes the output of all the neurons in this layer (for every instance in the mini-batch). 
##### Step 2:
* The result is passed on to the next layer, its output is computed and passed to the next layer, and so on until we get the output of the last layer, the output layer. 

**This is the forward pass:** it is exactly like making predictions, except all intermediate results are preserved since they are needed for the _backward pass_

#### Backward pass:
##### Step 1: Chain Rule
* After getting output of the output layer it computes how much each _output connection_ contributed to the error. This is done analytically by applying the **chain rule**, which makes this step fast and precise.
##### Step 2:
The algorithm then measures how much of these _error contributions came from each connection_ in the layer below, again using the **chain rule**, working _backward_ until the algorithm reaches the input layer. 

This reverse pass efficiently measures the error gradient across all the connection weights in the network by propagating the error gradient backward through the network (hence the name of the algorithm). Finally, the algorithm performs a Gradient Descent step to tweak all the connection weights in the network, using the error gradients it just computed.

### Techiniques Used
* Chain rule refresher
* Applying the chain rule
* Saving work with memoization

### Weight update:
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
For training of any neural network the aim is to minimize the loss (**_y - ŷ_**). The back propagation does this job by adjust each weight in the network in _proportion to how much it contributes to overall error_.
### The formula : <img src="https://latex.codecogs.com/svg.image?\large&space;\mathbf{W_{(new)}&space;=&space;W_{(old)}-\eta&space;{\color{Blue}&space;\frac{\partial&space;L&space;}{\partial&space;W_{(old)}}}}" title="https://latex.codecogs.com/svg.image?\large \mathbf{W_{(new)} = W_{(old)}-\eta {\color{Blue} \frac{\partial L }{\partial W_{(old)}}}}" align="center"/> <img src="https://latex.codecogs.com/svg.image?\large&space;\&space;\&space;\&space;\begin{cases}&space;{\color{Red}&space;\eta}&space;\mathrm{\&space;\&space;=&space;\&space;\&space;'eta'&space;\&space;is\&space;the\&space;learning\&space;rate&space;}&space;\&space;,\\&space;\\&space;{\color{Red}&space;\frac{\partial&space;L&space;}{\partial&space;W_{(old)}}&space;}&space;\mathrm{\&space;\&space;=&space;\&space;&space;&space;derivative\&space;of\&space;loss\&space;by\&space;derivative\&space;of\&space;old\&space;weight}\end{cases}&space;" title="https://latex.codecogs.com/svg.image?\large \ \ \ \begin{cases} {\color{Red} \eta} \mathrm{\ \ = \ \ 'eta' \ is\ the\ learning\ rate } \ ,\\ \\ {\color{Red} \frac{\partial L }{\partial W_{(old)}} } \mathrm{\ \ = \ derivative\ of\ loss\ by\ derivative\ of\ old\ weight}\end{cases} " align ="center"/>


* For the weight **'W4'** in the above diagram we just need to calculate <img src="https://latex.codecogs.com/svg.image?\large&space;\mathbf{W_{4(new)}&space;=&space;W_{4(old)}-\eta&space;{\color{black}&space;\frac{\partial&space;L&space;}{\partial&space;W_{4(old)}}}}" title="https://latex.codecogs.com/svg.image?\large \mathbf{W_{4(new)} = W_{4(old)}-\eta {\color{black} \frac{\partial L }{\partial W_{4(old)}}}}" width=25% align="center"/>

### But how does _`Derivative of Loss`_ by _`Derivative of old-weight`_  (  <img src="https://latex.codecogs.com/svg.image?\mathbf{{\color{Blue}&space;\frac{\partial&space;L&space;}{\partial&space;W_{(old)}}}}" title="https://latex.codecogs.com/svg.image?\mathbf{{\color{Blue} \frac{\partial L }{\partial W_{(old)}}}}" align="center"/>) come?
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)



<img src="https://user-images.githubusercontent.com/12748752/166604801-f3610f6b-c03d-4aa5-93c2-f225dcad2eaa.png" width=50% align="right"/>

#### For the -ve slope
* **W<sub>(new)</sub> = W<sub>(old)</sub> - &eta; (-ve)**
* **W<sub>(new)</sub> = W<sub>(old)</sub> +  &eta;**
* That makes it for always <img src="https://latex.codecogs.com/svg.image?\\&space;\mathbf{{\color{Red}&space;W_{(new)}&space;>&space;W_{(old)}}}"  align="center" />

#### For the +ve slope
* **W<sub>(new)</sub> = W<sub>(old)</sub> - &eta; (+ve)**
* **W<sub>(new)</sub> = W<sub>(old)</sub> -  &eta;**
* That makes it for always <img src="https://latex.codecogs.com/svg.image?\\&space;\mathbf{{\color{Red}&space;W_{(new)}&space;<&space;W_{(old)}}}" title="https://latex.codecogs.com/svg.image?\\ \mathbf{{\color{Red} W_{(new)} < W_{(old)}}}" align="center" />





<img src="https://user-images.githubusercontent.com/12748752/166584966-66a93072-31ab-4490-a934-e8a1b43eea55.png" width=60% />

#### Weight update:







