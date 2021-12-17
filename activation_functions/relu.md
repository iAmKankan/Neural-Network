## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

## ReLU (Rectified Linear Unit function)
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
* The Rectified Linear Unit function: ReLU(z) = max(0, z) 
* The ReLU function is continuous but unfortunately not differentiable at z = 0 (the slope changes abruptly, which can make Gradient Descent bounce around), and its derivative is 0 for z < 0. 
* In practice, however, it works very well and has the advantage of being fast to compute, so it has become the default. 
* Most importantly, the fact that it does not have a maximum output value helps reduce some issues during Gradient Descent.

![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
