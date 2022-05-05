## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

## Backpropagation

In 1986, David Rumelhart, Geoffrey Hinton, and Ronald Williams published a paper that introduced the **_backpropagation_** training algorithm. 

It is a technique for _computing the gradients automatically_ using _Gradient Descent_ in just two passes through the network (one **forward**, one **backward**), the backpropagation algorithm is able find out how each connection weight and each bias term should be tweaked in order to reduce the error. Once it has these gradients, it just performs a regular Gradient Descent step, and the whole process is repeated until the network converges to the solution.

<img src="https://user-images.githubusercontent.com/12748752/166812953-a0c9f542-1886-4f0c-bad4-a4cf815feeeb.png" width=70% align="center" />

It handles one **mini-batch** at a time (for example, containing 32 instances each), and it goes through the full training set multiple times. ( 1 pass = one epoch)
#### Step 1 forward pass:
* Each mini-batch is passed to the network’s **input layer**, which sends it to the **first hidden layer**. The algorithm then computes the output of all the neurons in this layer (for every instance in the mini-batch). 
* The result is passed on to the next layer, its output is computed and passed to the next layer, and so on until we get the output of the last layer, the output layer. 
* This is the forward pass: it is exactly like making predictions, except all intermediate results are preserved since they are needed for the backward pass.



## Backpropagation
* Chain rule refresher
* Applying the chain rule
* Saving work with memoization

The goals of backpropagation are straightforward: adjust each weight in the network in proportion to how much it contributes to overall error. If we iteratively reduce each weight’s error, eventually we’ll have a series of weights that produce good predictions.

### 1 Weight update:
For training of any neural network the aim is to minimize the loss (**_y - ŷ_**). The back propagation does this job by adjust each weight in the network in _proportion to how much it contributes to overall error_.
<img src="https://user-images.githubusercontent.com/12748752/166584966-66a93072-31ab-4490-a934-e8a1b43eea55.png" width=60% />
#### Weight update:
<img src="https://user-images.githubusercontent.com/12748752/166604801-f3610f6b-c03d-4aa5-93c2-f225dcad2eaa.png" width=50% />

#### The formula : <img src="https://latex.codecogs.com/svg.image?\mathbf{W_{(new)}&space;=&space;W_{(old)}-\eta&space;{\color{Blue}&space;\frac{\partial&space;L&space;}{\partial&space;W_{(old)}}}}" title="https://latex.codecogs.com/svg.image?\mathbf{W_{(new)} = W_{(old)}-\eta {\color{Blue} \frac{\partial L }{\partial W_{(old)}}}}" align="center"/>

<img src="https://latex.codecogs.com/svg.image?\\{\color{Red}&space;\eta}\&space;\&space;the\&space;'eta'&space;\&space;is\&space;the\&space;learning\&space;rate\&space;,\\&space;\\&space;{\color{Red}&space;\frac{\partial&space;L&space;}{\partial&space;W_{(old)}}&space;}\&space;is\&space;derivative\&space;of\&space;loss\&space;by\&space;derivative\&space;of\&space;old\&space;weight" title="https://latex.codecogs.com/svg.image?\\{\color{Red} \eta}\ \ the\ 'eta' \ is\ the\ learning\ rate\ ,\\ \\ {\color{Red} \frac{\partial L }{\partial W_{(old)}} }\ is\ derivative\ of\ loss\ by\ derivative\ of\ old\ weight" />

#### For the -ve slope
* **W<sub>(new)</sub> = W<sub>(old)</sub> - &eta; (-ve)**
* **W<sub>(new)</sub> = W<sub>(old)</sub> +  &eta;**
* That makes it for always <img src="https://latex.codecogs.com/svg.image?\\&space;\mathbf{{\color{Red}&space;W_{(new)}&space;>&space;W_{(old)}}}"  align="center" />

#### For the +ve slope
* **W<sub>(new)</sub> = W<sub>(old)</sub> - &eta; (+ve)**
* **W<sub>(new)</sub> = W<sub>(old)</sub> -  &eta;**
* That makes it for always <img src="https://latex.codecogs.com/svg.image?\\&space;\mathbf{{\color{Red}&space;W_{(new)}&space;<&space;W_{(old)}}}" title="https://latex.codecogs.com/svg.image?\\ \mathbf{{\color{Red} W_{(new)} < W_{(old)}}}" align="center" />


