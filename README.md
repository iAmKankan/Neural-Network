## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
### ◼️ _Neural-Network_
### ◼️ _Perceptron_
* [The Perceptron](https://github.com/iAmKankan/Neural-Network/blob/main/perceptron/README.md)
* [Multi-Layer Perceptron](https://github.com/iAmKankan/Neural-Network/blob/main/perceptron/README.md)
* [Training a Neural Network](https://github.com/iAmKankan/Neural-Network/blob/main/perceptron/README.md#training-perceptron)
### ◼️ [_Neural-Network Hyperparameters_](https://github.com/iAmKankan/Neural-Network/blob/main/hyperparameters/README.md)
   * [Activation Function](https://github.com/iAmKankan/Neural-Network/blob/main/activation_functions/README.md)
   * [Optimizers](https://github.com/iAmKankan/Neural-Network/tree/main/optimizer#readme)
### ◼️ _RNN, LSTM, GRU_
* [Recurrent Neural Networks](https://github.com/iAmKankan/Neural-Network/blob/main/rnn/README.md)
* Types of Recurrent Neural Networks
  * Simple RNN
  * LSTM
  * GRU
* Oparetions on RNN
   * [Sentiment Analysis](https://github.com/iAmKankan/Neural-Network/blob/main/sentiment.md)
 * Bidirectional Recurrent Neural Networks
### ◼️ _Natural Language Processing(NLP) with Deep Learning_
  * [Sequence to sequence models](https://github.com/iAmKankan/Neural-Network/tree/main/NLP#readme)
  * Encoder–Decoder Network for Neural Machine Translation
### ◼️ _Attention Mechanism_
   * [Attention](https://github.com/iAmKankan/Neural-Network/tree/main/Attention-Mechanisms#readme)
### ◼️ [_Neural-Network Common Terms_](https://github.com/iAmKankan/Neural-Network/blob/main/commonterms.md)
  * Neural Network
  * Neuron
  * Synapse
  * Weights 
  * Bias 
  * Layers 
  * Weighted Input 
  * Activation Functions 
  * Loss Functions 
  * Optimization Algorithms 
  * Gradient Accumulation 
  * Co-occurrence Matrix 
  * Negative sampling 
## ◼️ Software Engineering Vs AI/ML
<img src="https://user-images.githubusercontent.com/12748752/165293822-f6f8fe1c-ddd3-4ecb-8af9-12421f0d2639.png" width=50%/>

## ◼️ _Deep Learning & Neural-Network_
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
**_Deep learning_** implies the use of neural networks. The "**deep**" in deep learning refers to a _neural network_ with many _hidden layers_.
### _Feature Vector_
Neural networks accept input and produce output. The input to a neural network is called the _**feature vector**_. 
   * The size of this vector is always a **_fixed length_**. 
   * _Changing the size of the feature vector usually means recreating the entire neural network._ 
   * Though the feature vector is called a "vector", this is not always the case. A vector implies a **1D array**. In convolutional neural networks (CNNs), which can allow the input size to change without retraining the neural network. 

Historically the input to a neural network was always **1D**. However, with modern neural networks, you might see input data, such as:
  * **1D Vector** - Classic input to a neural network, similar to rows in a spreadsheet.  Common in predictive modeling.
  * **2D Matrix** - Grayscale image input to a convolutional neural network (CNN).
  * **3D Matrix** - Color image input to a convolutional neural network (CNN).
  * **nD Matrix** - Higher order input to a CNN.

### _Dimention_
The term **dimension** can be confusing in neural networks.  In the sense of a **1D** input vector, dimension refers to how many elements are in that 1D array.  
* **Example** a neural network with 10 input neurons has 10 dimensions.  
* However, now that we have CNN's, the input has dimensions too.  
* The input to the neural network will *usually* have 1, 2 or 3 dimensions.  4 or more dimensions is unusual.  
* You might have a 2D input to a neural network that has 64x64 pixels. 
* This would result in 4,096 input neurons.  
* This network is either** 2D or 4,096D, **depending on which set of dimensions you are talking about!**

### _Types of Neurons_
There are usually four types of neurons in a neural network:
  * **Input Neurons** - We map each input neuron to one element in the feature vector.
  * **Hidden Neurons** - Hidden neurons allow the neural network to be abstract and process the input into the output.
  * **Output Neurons** - Each output neuron calculates one part of the output.
  * **Bias Neurons** - Work similar to the y-intercept of a linear equation.

These neurons are grouped into layers:
