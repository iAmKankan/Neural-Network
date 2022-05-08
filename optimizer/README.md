## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* [**_Gradient Descent_**](https://github.com/iAmKankan/MachineLearning_With_Python/tree/master/training#readme)
* * [Gradient Descent](https://github.com/iAmKankan/MachineLearning_With_Python/blob/master/training/gradient_descent.md)
   * [Batch Gradient Descent](https://github.com/iAmKankan/MachineLearning_With_Python/blob/master/training/batch_gd.md)
   * [Stochastic Gradient Descent](https://github.com/iAmKankan/MachineLearning_With_Python/blob/master/training/stochastic_gradient_descent.md)


## Optimizer
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* In order to train a very large deep neural network and reach a better solution as well as speed up training we need so far: 
   * applying a good _`initialization strategy`_ for the connection weights, 
   * using a good _`activation function`_, 
   * using _`Batch Normalization`_, 
   * _`Transfer Learning`_ reusing parts of a pretrained network  (possibly built on an auxiliary task or using unsupervised learning).
*  Another huge speed boost comes from using a **_faster optimizer_** than the regular [**_Gradient Descent optimizer_**](https://github.com/iAmKankan/MachineLearning_With_Python/tree/master/training#readme) and its variants.
*  Most popular algorithms: 
   1. _**Momentum Optimization**_
   2. **_Nesterov Accelerated Gradient_**
   3. **_AdaGrad_**
   4. _**RMSProp**_  
   5. **_Adam and Nadam optimization_**

### Momentum Optimization
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

## Learning Rate
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
### Learning Rate Scheduling
Finding a good learning rate is very important. If you set it much too high, training may diverge (as we discussed in “Gradient Descent”). If you set it too low, training will eventually converge to the optimum, but it will take a very long time. If you set it slightly too high, it will make progress very quickly at first, but it will end up dancing around the optimum, never really settling down. If you have a limited computing budget, you may have to interrupt training before it has converged properly, yielding a suboptimal solution
