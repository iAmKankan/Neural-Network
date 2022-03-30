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

* Fortunately, there are many techniques to explore a search space much more efficiently than randomly. Their core idea is simple: when a region of the space turns out to be good, it should be explored more. Such techniques take care of the “zooming” process for you and lead to much better solutions in much less time. Here are some Python libraries you can use to optimize hyperparameters:









## References
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
