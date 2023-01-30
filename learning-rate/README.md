## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

## Learning Rate ( $\Large{\color{Purple}\mathbf{\eta}}$ ) Decay 
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
Most **Deep Learning techniques** tend to **decrease** the **learning rate** as the number of **iterations (epoch) increase**.
* High learning rates-parameter values shoot vary rapidly and by large amounts and would not settle down in a local minima
* A low learning rate would lead to small updates and convergence to a false minimum.


Neural network, is a very complicated function of the **weights**, it is very difficult to say when we are **near optimal minima** and when we are **near some very poor minima**, so that must be some systematic way of changing the **learning rate**, this is basically your $\Large{\color{Purple}\mathbf{\eta}}$ , as we have seen ${\color{Purple}\mathbf{x=}\Large \mathrm{\eta\frac{\partial L}{\partial x}}}$, or in case of weights ${\color{Purple}\mathbf{W=}\Large \mathrm{ \eta\frac{\partial L}{\partial W}}}$ .

This learning rate will dictate _by how much to change your parameter weights_ , Because the **magnitude of the update** (during the backprop) also depends not only on the **gradient of the loss function with respect to the weights** $\Large{\color{Purple}\frac{\partial L}{\partial W}}$ , but also on the learning rate  $\Large{\color{Purple}\mathbf{\eta}}$ .

So, by **modulating** $\Large{\color{Purple}\frac{\partial L}{\partial W}}$ you can also **modulate** the **magnitude of your updates**.

Okay, so what happens is that when you have **very high learning rates**, so let say $\Large{\color{Purple}\mathbf{\eta}}$ is **very large number**, let say close to **1**, then your parameter values **shoot** vary **rapidly** and by **large amounts** and would _not settle down_ in **local minima**.

However **lower learning rate** would lead to **slow learning** which means parameters will not change rapidly and it is quite possible that they can stuck in, some false minima. 


So then this, there is no good way to know when to do what but typically there are techniques that people use is to **decrease the learning rate** as the number of iterations or epochs increase.So there are many ways to do that-

##### 1. Reduce the **learning rate** by **a constant factor** in every epoch or in every _k_ epoch: 

##### 2. Alternatively, you can check the validation error and reduce the learning rate by a factor _k_ every time the _validation error drops_:


##### 3. Another way for smooth decrease in learning rate over epoch we use-

$$ \Large{\color{Purple} \mathrm{\eta} = \mathrm{\frac{\eta_0}{1 + decay * epoch}}}$$

* $\Large{\color{Purple}\mathbf{\eta_0}}$ is your initial learning rate, which you set, this is another **hyper parameter**.
* Divided by **1** plus there is some **decay rate** times the **epoch number**. 

So, as the number of **epoch increase** depending on the **magnitude** of the **decay**, your initial learning rate will also decrease.  so this a very smooth way of decreasing your learning rate.
