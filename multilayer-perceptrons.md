## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

## Multi-Layer Perceptron
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* An MLP is composed of -
   * **One (passthrough) input layer,**
   * **One or more layers of TLUs (Threshold Logic Unit)- called hidden layers,**
   * **One final layer of TLU (Threshold Logic Unit) called the output layer.**

* The layers close to the input layer are usually called the **lower layers**, and the ones close to the outputs are usually called the **upper layers**. 
* Every layer except the output layer includes **a bias neuron** and is fully connected to the next layer.

> #### The signal flows only in one direction (from the inputs to the outputs), so this architecture is an example of a **Feedforward Neural Network (FNN)**.

<img src="https://user-images.githubusercontent.com/12748752/143045465-2fe26cb7-48ea-4590-b381-24215f014004.png" width=30% />


> #### When an ANN contains a deep stack of hidden layers, it is called a Deep Neural Network (DNN).

## Backpropagation 
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
*  **Backpropagation is simply Gradient Descent using an efficient technique for computing the gradients automatically**: 

* In just two passes through the network (one **forward**, one **backward**), the backpropagation algorithm is able to compute **the gradient of the network’s error** with regards to every single model parameter. 

> ### In other words, it can find out how each connection weight and each bias term should be tweaked in order to reduce the error. 
* Once it has these gradients, it just performs a regular Gradient Descent step, and the whole process is repeated until the network converges to the solution.

> #### Automatically computing gradients is called *automatic differentiation*, or *autodiff*. The autodiff technique used by backpropagation is called *reverse-mode autodiff*. It is fast and precise, and is well suited when the function to differentiate has many variables (e.g., connection weights) and few outputs (e.g., one loss). 


## Bibliography

* **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition by Aurélien Géron**
