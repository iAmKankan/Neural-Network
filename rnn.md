## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

## Recurrent Neural Networks(RNN)
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
> #### Feedforward Neural Networks
* Where the activations flow only in one direction, from the _`input layer`_ to the _`output layer`_ (a few exceptions are there). 

> #### Recurrent Neural Network(RNN)
* It looks very much like a feedforward neural network, except it also has connections pointing backward. 

<img src="https://user-images.githubusercontent.com/12748752/144027506-ea345023-d777-493c-88fd-fae058e70097.png" width=50%/>

> ####  <ins>A recurrent neuron (left) unrolled through time (right)</ins>

* Let’s look at the simplest possible RNN, composed of one neuron receiving inputs, producing an output, and sending that output back to itself, as shown in Figure 15-1 (left). At each time step t (also called a frame), this recurrent neuron receives the inputs x as well as its own output from the previous time step, y . Since there is no previous output at the first time step, it is generally set to 0. We can represent this tiny network against the time axis, as shown in Figure 15-1 (right). This is called unrolling the network through time (it’s the same recurrent neuron represented once per time step).
* 












## Bibliography
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition by Aurélien Géron**
