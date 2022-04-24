## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)


## Fine-Tuning Neural Network Hyperparameters
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
**The flexibility of neural networks is also one of their main drawbacks:** there are many hyperparameters to tweak. Not only can you use any imaginable network architecture, but even in a simple MLP you can change the number of layers, **_the number of neurons per layer_**, _**the type of activation function to use in each layer**_, **_the weight initialization logic_**, and much more. 

_How do you know what combination of hyperparameters is the best for your task?_

#### Solution # 1:
* Simply try many combinations of hyperparameters and see which one works best on the validation set (or use K-fold crossvalidation). To do this, we need to wrap our Keras models in objects that mimic regular Scikit-Learn regressors.
  * _`GridSearchCV`_ : 
  * _`RandomizedSearchCV`_
#### Problems:
* **_GridSearchCV_**:We don’t want to train and evaluate a single model like this, though we want to train hundreds of variants and see which one performs best on the validation set. [_Note that the **score** will be the opposite of the **MSE** because Scikit-Learn wants scores, not losses (i.e., higher should be better)._]
* **_RandomizedSearchCV_**:Using randomized search is not too hard, and it works well for many fairly simple problems. When training is slow, however (e.g., for more complex problems with larger datasets), this approach will only explore a tiny portion of the hyperparameter space. [_Note that **RandomizedSearchCV** uses **K-fold crossvalidation**, so it does not use **X_valid** and **y_valid**, which are only used for **early stopping**._]

#### Solution #2:
* Fortunately, there are many techniques to explore a search space much more efficiently than randomly. Their core idea is simple: when a region of the space turns out to be good, it should be explored more. Such techniques take care of the “zooming” process for you and lead to much better solutions in much less time. Here are some Python libraries you can use to optimize hyperparameters:
  * Hyperopt
  * Hyperas, kopt, or Talos
  * Keras Tuner
  * Scikit-Optimize (skopt)
  * Spearmint
  * Hyperband
  * Sklearn-Deap

But despite all this exciting progress and all these tools and services, it still helps to have an idea of what values are reasonable for each hyperparameter so that you can build a quick prototype and restrict the search space. The following sections provide guidelines for choosing the number of hidden layers and neurons in an MLP and for selecting good values for some of the main hyperparameters.

### _Number of Hidden Layers_
For many problems, you can begin with a single hidden layer and get reasonable results. An **Multi-Layered Perceptron(MLP)** with just one hidden layer can theoretically model even the most complex functions, provided it has enough neurons. 
#### Why we need Deep Networks?
_For complex problems, deep networks have a much higher parameter efficiency than shallow ones: they can model complex functions using exponentially fewer neurons than shallow nets, allowing them to reach much **better performance with the same amount of training data**._ 
* Suppose you are asked to draw a forest using some drawing software, but forbidden to copy and paste anything. If you could draw one leaf, copy and paste it to draw a branch, then copy and paste that branch to create a tree, and finally copy and paste this tree to make a forest, you would be finished in no time. 
* Real-world data is often structured in such a _hierarchical way_, and deep neural networks automatically take advantage of this fact: 
  * lower hidden layers model lowlevel structures (e.g., line segments of various shapes and orientations), 
  * intermediate hidden layers combine these low-level structures to model intermediate-level structures (e.g., squares, circles), and 
  * the highest hidden layers and the output layer combine these intermediate structures to model high-level structures (e.g., faces).
  * but it also improves their ability to generalize to new datasets.
* _For complex problems, you can ramp up the number of hidden layers until you start overfitting the training set._ 
* Very complex tasks, such as large image classification or speech recognition, typically require networks with dozens of layers (or even hundreds, but not fully connected ones), and they need a huge amount of training data. 
* You will rarely have to train such networks from scratch: it is much more common to reuse parts of a pretrained state-of-the-art network that performs a similar task.
* **_Transfer Learning_** This way the network will not have to learn from scratch all the low-level structures that occur in most pictures; it will only have to learn the higher-level structures.

### _Number of Neurons per Hidden Layer_
The number of neurons in the input and output layers is determined by the type of input and output your task requires. For example, the MNIST task requires 28 × 28 = 784 input neurons and 10 output neurons.

## Architecting Network: Hyperparameters
**_The number of layers_**, **_neuron counts per layers_**, **_layer types_** and **_activation functions_** are all choices you must make to optimize your neural network.  Some of the categories of hyperparameters for you to choose from come from the following list:

* Number of Hidden Layers and Neuron Counts
* Activation Functions
* Advanced Activation Functions
* Regularization: L1, L2, Dropout
* Batch Normalization
* Training Parameters

The following sections will introduce each of these categories for Keras. While I will provide you with some general guidelines for hyperparameter selection; no two tasks are the same.  You will benefit from experimentation with these values to determine what works best for your neural network.  In the next part, we will see how machine learning can select some of these values on its own.






## References
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
