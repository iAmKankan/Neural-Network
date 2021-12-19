## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* [Initialization](#)
* [Activation Function](https://github.com/iAmKankan/Neural-Network/blob/main/activation_functions/README.md)
* [Why we need Batch Normalization](url)
   * [Batch Normalization](url)
* [Gradient Clipping](#gradient-clipping)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

## Why we need Batch Normalization
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* In order to resolve the _**vanishing/exploding gradients**_ problem if we use **_`He initialization`_** along with **_ELU_** (or any variant of ReLU) can significantly reduce the problems at the beginning of training, it doesn’t guarantee that they won’t come back during training.
## Batch Normalization
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* _**Batch Normalization**_ consists of adding an _operation_ in the model _just before_ or _after_ the **_activation function_** of each hidden layer. 
* This **operation** simply 
   * **zerocenters and normalizes** each input, 
   * then **scales** and **shifts** the result using two new parameter vectors per layer: one for _scaling_, the other for _shifting_. 
* In other words, the operation lets the model learn the optimal scale and mean of each of the layer’s inputs. 
* In many cases, if you add a BN layer as the very first layer of your neural network, you do not need to standardize your training set (e.g., using a StandardScaler); 
* The BN layer will do it for you (well, approximately, since it only looks at one batch at a time, and it can also rescale and shift each input feature).
* In order to _**zero-center**_ and **_normalize_** the inputs, the algorithm estimates each input’s **_mean_** and **_standard deviation_** of the input over the current **_mini-batch_** (hence the name _`“Batch Normalization”`_).
> #### Finally, like a gift that keeps on giving, Batch Normalization acts like a regularizer, reducing the need for other regularization techniques (such as dropout).

## Gradient Clipping
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* Gradient Clipping is a technique to mitigate the **_exploding gradients_** problem is to clip the gradients during backpropagation so that they never exceed some threshold. 
* This technique is most often used in recurrent neural networks, as **Batch Normalization** is tricky to use in **RNNs**. 
* For other types of networks, Batch Normalization is usually sufficient.
* In Keras, implementing Gradient Clipping is just a matter of setting the _`clipvalue`_ or _`clipnorm`_ argument when creating an optimizer, like this:
```Python
optimizer = keras.optimizers.SGD(clipvalue=1.0)
model.compile(loss="mse", optimizer=optimizer)
```

