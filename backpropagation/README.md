## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)


## Backpropagation
* Chain rule refresher
* Applying the chain rule
* Saving work with memoization

The goals of backpropagation are straightforward: adjust each weight in the network in proportion to how much it contributes to overall error. If we iteratively reduce each weight’s error, eventually we’ll have a series of weights that produce good predictions.

### 1 Weight update:
For training of any neural network the aim is to minimize the loss (**_y - ŷ_**). The back propagation does this job by adjust each weight in the network in _proportion to how much it contributes to overall error_.
<img src="https://user-images.githubusercontent.com/12748752/166584966-66a93072-31ab-4490-a934-e8a1b43eea55.png" width=60% />
#### Weight update formula:
<img src="https://latex.codecogs.com/svg.image?\mathbf{W_{(new)}&space;=&space;W_{(old)}-\eta&space;{\color{Blue}&space;\frac{\partial&space;h&space;}{\partial&space;W_{(old)}}}}" title="https://latex.codecogs.com/svg.image?\mathbf{W_{(new)} = W_{(old)}-\eta {\color{Blue} \frac{\partial h }{\partial W_{(old)}}}}" />

<img src="https://latex.codecogs.com/svg.image?{\color{Red}&space;\eta}\&space;\&space;is\&space;the\&space;learning\&space;rate\&space;,\&space;{\color{Red}&space;\frac{\partial&space;h&space;}{\partial&space;W_{(old)}}&space;}\&space;is\&space;derivative\&space;of\&space;loss\&space;or\&space;the\&space;slope" title="https://latex.codecogs.com/svg.image?{\color{Red} \eta}\ \ is\ the\ learning\ rate\ ,\ {\color{Red} \frac{\partial h }{\partial W_{(old)}} }\ is\ derivative\ of\ loss\ or\ the\ slope" />

<img src="https://user-images.githubusercontent.com/12748752/166604801-f3610f6b-c03d-4aa5-93c2-f225dcad2eaa.png" width=50% />
