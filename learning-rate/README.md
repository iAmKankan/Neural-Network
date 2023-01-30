## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

## Learning Rate ( $\Large{\color{Purple}\mathbf{\eta}}$ ) Decay 
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
Most **Deep Learning techniques** tend to **decrease** the **learning rate** as the number of **iterations (epoch) increase**.
* High learning rates-parameter values would vary rapidly and by large amounts and would not settle down in a local minima
* A low learning rate would lead to small updates and convergence to a false minimum.


Neural network, is a very complicated function of the **weights**, it is very difficult to say when we are **near optimal minima** and when we are **near some very poor minima**, so that must be some systematic way of changing the **learning rate**, this is basically your $\Large{\color{Purple}\mathbf{\eta}}$ , as we have seen $\Large{\color{Purple}\mathbf{x= \eta\frac{\partial L}{\partial x}}}$, or in case of weights $\Large{\color{Purple}\mathbf{W= \eta\frac{\partial L}{\partial W}}}$ .
