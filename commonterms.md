## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
### ◼️ _Neural-Network Common Terms_
  * [Neural Network](#neural-network)
  * [Neuron](#neuron)
  * [Synapse](#synapse)
  * [Weights](#weights)
  * [Bias](#bias)
  * [Layers](#layers)
  * [Weighted Input](#weighted-input)
  * [Activation Functions](#activation-functions)
  * [Loss Functions](#loss-functions)
  * [Optimization Algorithms](#optimization-algorithms)
  * [Gradient Accumulation](#gradient-accumulation)
  * [Co-occurrence Matrix](#co-occurrence-matrix)
  * [Negative sampling](#negative-sampling)
  * [Time series](#time-series)
##  Neural-Network _Common Terms_
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
### _Neural Network_
Neural networks are a class of machine learning algorithms used to model complex patterns in datasets using multiple hidden layers and non-linear activation functions. A neural network takes an input, passes it through multiple layers of hidden neurons (mini-functions with unique coefficients that must be learned), and outputs a prediction representing the combined input of all the neurons.

Neural networks are trained iteratively using optimization techniques like gradient descent. After each cycle of training, an error metric is calculated based on the difference between prediction and target. The derivatives of this error metric are calculated and propagated back through the network using a technique called backpropagation. Each neuron’s coefficients (weights) are then adjusted relative to how much they contributed to the total error. This process is repeated iteratively until the network error drops below an acceptable threshold.
### Neuron
A neuron takes a group of weighted inputs, applies an activation function, and returns an output.
Inputs to a neuron can either be features from a training set or outputs from a previous layer’s neurons. Weights are applied to the inputs as they travel along synapses to reach the neuron. The neuron then applies an activation function to the “sum of weighted inputs” from each incoming synapse and passes the result on to all the neurons in the next layer.
### Synapse
Synapses are like roads in a neural network. They connect inputs to neurons, neurons to neurons, and neurons to outputs. In order to get from one neuron to another, you have to travel along the synapse paying the “toll” (weight) along the way. Each connection between two neurons has a unique synapse with a unique weight attached to it. When we talk about updating weights in a network, we’re really talking about adjusting the weights on these synapses.
### Weights
Weights are values that control the strength of the connection between two neurons. That is, inputs are typically multiplied by weights, and that defines how much influence the input will have on the output. In other words: when the inputs are transmitted between neurons, the weights are applied to the inputs along with an additional value (the bias)
### Bias
Bias terms are additional constants attached to neurons and added to the weighted input before the activation function is applied. Bias terms help models represent patterns that do not necessarily pass through the origin. For example, if all your features were 0, would your output also be zero? Is it possible there is some base value upon which your features have an effect? Bias terms typically accompany weights and must also be learned by your model.
### Layers
#### Input Layer
Holds the data your model will train on. Each neuron in the input layer represents a unique attribute in your dataset (e.g. height, hair color, etc.).
#### Hidden Layer
Sits between the input and output layers and applies an activation function before passing on the results. There are often multiple hidden layers in a network. In traditional networks, hidden layers are typically fully-connected layers — each neuron receives input from all the previous layer’s neurons and sends its output to every neuron in the next layer. This contrasts with how convolutional layers work where the neurons send their output to only some of the neurons in the next layer.
#### Output Layer
The final layer in a network. It receives input from the previous hidden layer, optionally applies an activation function, and returns an output representing your model’s prediction.
### Weighted Input
A neuron’s input equals the sum of weighted outputs from all neurons in the previous layer. Each input is multiplied by the weight associated with the synapse connecting the input to the current neuron. If there are 3 inputs or neurons in the previous layer, each neuron in the current layer will have 3 distinct weights — one for each each synapse.
#### Single Input
<img src="https://latex.codecogs.com/svg.image?\\Z&space;=&space;Input.Weight\\Z&space;&space;=&space;XW" title="https://latex.codecogs.com/svg.image?\\Z = Input.Weight\\Z = XW" />

#### Multiple Inputs
<img src="https://latex.codecogs.com/svg.image?\\Z&space;=&space;\sum_{i=1}^{n}x_iw_i\\Z&space;=&space;x_1w_1&plus;x_2w_2&plus;x_3w_3" title="https://latex.codecogs.com/svg.image?\\Z = \sum_{i=1}^{n}x_iw_i\\Z = x_1w_1+x_2w_2+x_3w_3" />
Notice, it’s exactly the same equation we use with linear regression! In fact, a neural network with a single neuron is the same as linear regression! The only difference is the neural network post-processes the weighted input with an activation function.

### Activation Functions
Activation functions live inside neural network layers and modify the data they receive before passing it to the next layer. Activation functions give neural networks their power — allowing them to model complex non-linear relationships. By modifying inputs with non-linear functions neural networks can model highly complex relationships between features. Popular activation functions include relu and sigmoid.
Activation functions typically have the following properties:
 * **Non-linear** - In linear regression we’re limited to a prediction equation that looks like a straight line. This is nice for simple datasets with a one-to-one relationship between inputs and outputs, but what if the patterns in our dataset were non-linear? (e.g. x2, sin, log). To model these relationships we need a non-linear prediction equation.¹ Activation functions provide this non-linearity.
 * **Continuously differentiable** — To improve our model with gradient descent, we need our output to have a nice slope so we can compute error derivatives with respect to weights. If our neuron instead outputted 0 or 1 (perceptron), we wouldn’t know in which direction to update our weights to reduce our error.
 * **Fixed Range** — Activation functions typically squash the input data into a narrow range that makes training the model more stable and efficient.

### Loss Functions
A loss function, or cost function, is a wrapper around our model’s predict function that tells us “how good” the model is at making predictions for a given set of parameters. The loss function has its own curve and its own derivatives. The slope of this curve tells us how to change our parameters to make the model more accurate! We use the model to make predictions. We use the cost function to update our parameters. Our cost function can take a variety of forms as there are many different cost functions available. Popular loss functions include: MSE (L2) and Cross-entropy Loss.

### Optimization Algorithms
* [Link](https://github.com/iAmKankan/Neural-Network/tree/main/optimizer)
### Gradient Accumulation
Gradient accumulation is a mechanism to split the batch of samples—used for training a neural network—into several mini-batches of samples that will be run sequentially.

This is used to enable using large batch sizes that require more GPU memory than available. Gradient accumulation helps in doing so by using mini-batches that require an amount of GPU memory that can be satisfied.

Gradient accumulation means running all mini-batches sequentially (generally on the same GPU) while accumulating their calculated gradients and not updating the model variables - the weights and biases of the model. The model variables must not be updated during the accumulation in order to ensure all mini-batches use the same model variable values to calculate their gradients. Only after accumulating the gradients of all those mini-batches will we generate and apply the updates for the model variables.

### _Co-occurrence Matrix_
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* Generally speaking, a co-occurrence matrix will have specific entities in rows (**ER**) and columns (**EC**). 
* The purpose of this matrix is to present the number of times each **ER** appears in the same context as each **EC**. 
* As a consequence, in order to use a co-occurrence matrix, you have to define your entites and the context in which they co-occur.
* In NLP, the most classic approach is to define each entity (ie, lines and columns) as a word present in a text, and the context as a sentence.

<img src="http://latex.codecogs.com/svg.latex?\begin{matrix}&space;&\mathrm{aardvark}&&space;\mathrm{\cdots}&space;&&space;\mathrm{computer}&space;&&space;\mathrm{data}&space;&&space;\mathrm{result}&space;&\mathrm{&space;pie}&&space;\mathrm{sugar}&space;\\\mathrm{cherry}&space;&&space;0&space;&\cdots&&space;2&space;&&space;8&space;&&space;9&&space;442&&space;25\\\mathrm{strawberry}&&space;0&space;&\cdots&&space;0&space;&&space;0&space;&&space;1&space;&&space;60&&space;19\\\mathrm{digital}&&space;{\color{Red}&space;0}&space;&{\color{Red}\cdots}&&space;{\color{Red}1670}&space;&&space;{\color{Red}1683}&space;&&space;{\color{Red}85}&&space;{\color{Red}&space;5}&&space;{\color{Red}4}\\\mathrm{information}&&space;0&space;&\cdots&&space;3325&&space;3982&&space;378&&space;5&&space;13\\\end{matrix}&space;" title="http://latex.codecogs.com/svg.latex?\begin{matrix} &\mathrm{aardvark}& \mathrm{\cdots} & \mathrm{computer} & \mathrm{data} & \mathrm{result} &\mathrm{ pie}& \mathrm{sugar} \\\mathrm{cherry} & 0 &\cdots& 2 & 8 & 9& 442& 25\\\mathrm{strawberry}& 0 &\cdots& 0 & 0 & 1 & 60& 19\\\mathrm{digital}& {\color{Red} 0} &{\color{Red}\cdots}& {\color{Red}1670} & {\color{Red}1683} & {\color{Red}85}& {\color{Red} 5}& {\color{Red}4}\\\mathrm{information}& 0 &\cdots& 3325& 3982& 378& 5& 13\\\end{matrix} " />

#### What are they used for in NLP ?
* The most evident use of these matrix is their ability to provide links between notions. 
* Let's suppose you're working on products reviews. 
* Let's also suppose for simplicity that each review is only composed of short sentences. 
* You'll have something like that :
```
ProductX is amazing.
I hate productY.
```
* Representing these reviews as one co-occurrence matrix will enable you associate products with appreciations.

### Coreference Resolution
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* _Coreference resolution is the task of finding all expressions that refer to the same entity in a text._ 
* It is an important step for a lot of higher level NLP tasks that involve natural language understanding such as document summarization, question answering, and information extraction.
<img src="https://user-images.githubusercontent.com/12748752/161392419-c0339364-baa1-4600-ac6b-4c486964e213.png" width=40%/>

### Time Series
Suppose you are studying the number of _active users per hour on your website_, or _the daily temperature in your city_, or _your company’s financial health_, measured quarterly using multiple metrics. 

In all these cases, the data will be a sequence of one or more values per time step. This is called a time series. 
   1) In the first two examples there is _a single value per time step_, so these are **_univariate time series_**, 
   2) In the financial example there are _multiple values per time step_ (e.g., the company’s revenue, debt, and so on), so it is a **_multivariate time series_**. 
 
> #### A typical task is to predict future values, which is called **_forecasting_**. 
> #### Another common task is to fill in the blanks: to predict (or rather “postdict”) missing values from the past. This is called **_imputation_**.

### Forwardpropagation
#### Simple Network:
<img src="https://user-images.githubusercontent.com/12748752/165742485-709a249f-8fad-46aa-ab6b-9f49417b43ae.png" />
Forward propagation is how neural networks make predictions. Input data is "forward propagated" through the network layer by layer to the final layer which outputs a prediction. For the toy neural network above, a single pass of forward propagation translates mathematically to:
<p align="center"><b><i> Prediction = A( A( XW<sub>h</sub>) W<sub>o</sub>)</b></i></p>

### Context
* Meaning at any point of a sequence.

## Bibliography
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition by Aurélien Géron**
