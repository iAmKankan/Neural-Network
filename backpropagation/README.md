## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

<img src="https://latex.codecogs.com/svg.image?\frac{d}{dx}x^n&space;=&space;nx^{n-1}" title="\frac{d}{dx}x^n = nx^{n-1}" />

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


