## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* [Multi-Layer Perceptron](#multi-layer-perceptron)
* [Backpropagation](#backpropagation)
  * [How Backpropagation works](#how-backpropagation-works)

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
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
*  **Backpropagation is simply Gradient Descent using an efficient technique for computing the gradients automatically**: 

* In just two passes through the network (one **forward**, one **backward**), the backpropagation algorithm is able to compute **the gradient of the network’s error** with regards to every single model parameter. 

> ### In other words, it can find out how each connection weight and each bias term should be tweaked in order to reduce the error. 
* Once it has these gradients, it just performs a regular Gradient Descent step, and the whole process is repeated until the network converges to the solution.

> #### Automatically computing gradients is called *automatic differentiation*, or *autodiff*. The autodiff technique used by backpropagation is called *reverse-mode autodiff*. It is fast and precise, and is well suited when the function to differentiate has many variables (e.g., connection weights) and few outputs (e.g., one loss). 

### How Backpropagation works
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

* It handles one mini-batch at a time (for example containing 32 instances each), and it goes through the full training set multiple times. Each pass is called an **epoch**. 
* Each mini-batch is passed to the network’s input layer, which just sends it to the first hidden layer. The algorithm then computes the output of all the neurons in this layer (for every instance in the mini-batch). The result is passed on to the next layer, its output is computed and passed to the next layer, and so on until we get the output of the last layer, the output layer. This is the forward pass: it is exactly like making predictions, except all intermediate results are preserved since they are needed for the backward pass. 
* Next, the algorithm measures the network’s output error (i.e., it uses a loss function that compares the desired output and the actual output of the network, and returns some measure of the error). 
* Then it computes how much each output connection contributed to the error. This is done analytically by simply applying the [chain rule](https://github.com/iAmKankan/Mathematics/blob/main/D_calculus.md#chain-rule) (perhaps the most fundamental rule in calculus), which makes this step fast and precise. 
* The algorithm then measures how much of these error contributions came from each connection in the layer below, again using the [chain rule](https://github.com/iAmKankan/Mathematics/blob/main/D_calculus.md#chain-rule) —and so on until the algorithm reaches the input layer. As we explained earlier, this reverse pass efficiently measures the error gradient across all the connection weights in the network by propagating the error gradient backward through the network (hence the name of the algorithm).
* Finally, the algorithm performs a Gradient Descent step to tweak all the connection weights in the network, using the error gradients it just computed.

> ### Summary:
> * For each training instance the backpropagation algorithm goes steps like-
>   * **First makes a prediction (forward pass),**
>   * **Measures the error,**
>   * **Then goes through each layer in reverse to measure the error contribution from each connection (reverse pass),** 
>   * **Finally slightly tweaks the connection weights to reduce the error (Gradient Descent step).**

---
> ### It is important to initialize all the hidden layers’ connection weights randomly. 
> * For example, if you initialize all weights and biases to zero, then all neurons in a given layer will be perfectly identical, and thus backpropagation will affect them in exactly the same way, so they will remain identical. 
> * In other words, despite having hundreds of neurons per layer, your model will act as if it had only one neuron per layer: it won’t be too smart. 
> * If instead you randomly initialize the weights, you break the symmetry and allow backpropagation to train a diverse team of neurons.


## Regression MLPs
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

* First, MLPs can be used for regression tasks. 
  * If you want to predict a single value (e.g., the price of a house given many of its features), then you just need a single output neuron: its output is the predicted value.
  * For multivariate regression (i.e., to predict multiple values at once), you need one output neuron per output dimension. 
    * For example, to locate the center of an object on an image, you need to predict 2D coordinates, so you need two output neurons. 
    * If you also want to place a bounding box around the object, then you need two more numbers: the width and the height of the object.
    *  So you end up with 4 output neurons.

> #### In general, when building an MLP for regression, you do not want to use any activation function for the output neurons, so they are free to output any range of values.

> ### Typical Regression MLP Architecture

<img src="https://latex.codecogs.com/svg.image?\begin{matrix}\\\textbf{Hyperparameter}&space;&&space;\textbf{Typical&space;Value}\\\mathrm{input\&space;neurons}&space;&&space;\mathrm{One\&space;per\&space;input\&space;feature\&space;(e.g.,\&space;28\&space;x\&space;28\&space;=\&space;784\&space;for\&space;MNIST)}\\\mathrm{hidden\&space;layers}&space;&&space;\mathrm{Depends\&space;on\&space;the\&space;problem.\&space;Typically\&space;1\&space;to\&space;5.}\\\mathrm{neurons\&space;per\&space;hidden\&space;layer}&space;&&space;\mathrm{Depends\&space;on\&space;the\&space;problem.\&space;Typically\&space;10\&space;to\&space;100.}\\\mathrm{output\&space;neurons}&space;&&space;\mathrm{1\&space;per\&space;prediction\&space;dimension}\\\mathrm{Hidden\&space;activation}&space;&&space;\mathrm{ReLU\&space;(or\&space;SELU,\&space;see\&space;Chapter\&space;11)}\\\mathrm{Output\&space;activation}&space;&&space;\mathrm{None\&space;or\&space;ReLU/Softplus\&space;(if\&space;positive\&space;outputs)\&space;or\&space;Logistic/Tanh\&space;(if\&space;bounded\&space;outputs)}\\\mathrm{Loss\&space;function}&space;&&space;\mathrm{MSE\&space;or\&space;MAE/Huber\&space;(if\&space;outliers)}\\\end{matrix}" title="\begin{matrix}\\\textbf{Hyperparameter} & \textbf{Typical Value}\\\mathrm{input\ neurons} & \mathrm{One\ per\ input\ feature\ (e.g.,\ 28\ x\ 28\ =\ 784\ for\ MNIST)}\\\mathrm{hidden\ layers} & \mathrm{Depends\ on\ the\ problem.\ Typically\ 1\ to\ 5.}\\\mathrm{neurons\ per\ hidden\ layer} & \mathrm{Depends\ on\ the\ problem.\ Typically\ 10\ to\ 100.}\\\mathrm{output\ neurons} & \mathrm{1\ per\ prediction\ dimension}\\\mathrm{Hidden\ activation} & \mathrm{ReLU\ (or\ SELU,\ see\ Chapter\ 11)}\\\mathrm{Output\ activation} & \mathrm{None\ or\ ReLU/Softplus\ (if\ positive\ outputs)\ or\ Logistic/Tanh\ (if\ bounded\ outputs)}\\\mathrm{Loss\ function} & \mathrm{MSE\ or\ MAE/Huber\ (if\ outliers)}\\\end{matrix}" />


<img src="https://user-images.githubusercontent.com/12748752/143228831-e4318e6f-25b0-43b9-950d-a865c4df7d1c.png" width=60% height=30% />

## Classification MLPs
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

> ### Typical Classification MLP Architecture

<img src="https://latex.codecogs.com/svg.image?\begin{matrix}\\\textbf{Hyperparameter}&space;&&space;\textbf{Binary\&space;classification}&&space;\textbf{Multilabel\&space;binary\&space;classification}&&space;\textbf{Multiclass\&space;classification}\\\mathrm{Input\&space;and\&space;hidden\&space;layers}&space;&&space;\mathrm{Same\&space;as\&space;regression}&space;&&space;\mathrm{Same\&space;as\&space;regression}&space;&&space;\mathrm{Same\&space;as\&space;regression}\\\mathrm{output\&space;neurons}&&space;1&space;&&space;\mathrm{1\&space;per\&space;label}&space;&&space;\mathrm{1\&space;per\&space;class}\\&space;\mathrm{Output\&space;layer\&space;activation}&&space;\mathrm{Logistic}&space;&&space;\mathrm{Logistic}&space;&&space;\mathrm{Softmax}\\\mathrm{Loss\&space;function}&space;&&space;\mathrm{Cross-Entropy}&space;&&space;\mathrm{Cross-Entropy}&space;&&space;\mathrm{Cross-Entropy}\end{matrix}" title="\begin{matrix}\\\textbf{Hyperparameter} & \textbf{Binary\ classification}& \textbf{Multilabel\ binary\ classification}& \textbf{Multiclass\ classification}\\\mathrm{Input\ and\ hidden\ layers} & \mathrm{Same\ as\ regression} & \mathrm{Same\ as\ regression} & \mathrm{Same\ as\ regression}\\\mathrm{output\ neurons}& 1 & \mathrm{1\ per\ label} & \mathrm{1\ per\ class}\\ \mathrm{Output\ layer\ activation}& \mathrm{Logistic} & \mathrm{Logistic} & \mathrm{Softmax}\\\mathrm{Loss\ function} & \mathrm{Cross-Entropy} & \mathrm{Cross-Entropy} & \mathrm{Cross-Entropy}\end{matrix}" />


## Bibliography
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition by Aurélien Géron**
