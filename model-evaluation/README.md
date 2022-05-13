## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

## Model Performance Evaluation - Classification
#### 1) Confusion Matrix:
#### 2) Gain and Lift Charts
#### 3) Lift Chart
#### 4) K-S Chart 
#### 5) ROC Chart
#### 6) Area Under the Curve (AUC)


## 1) Confusion Matrix:
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
<img src="https://user-images.githubusercontent.com/12748752/168250710-596bf1a2-5d86-464d-b07f-70264ce4a3c3.png" width=30%/>
#### <ins><i>Confusion Matrix</i></ins>
An NxN table that aggregates a classification model's correct and incorrect guesses. One axis of a confusion matrix is the label that the model predicted, and the other axis is the ground truth. N represents the number of classes. For example, N=2 for a binary classification model. For example, here is a sample confusion matrix for a binary classification model:

### ***Classification reports***
- Precision Score
- Recall Score
- F1 Score
- Support
- ConfusionMatrix

### Evaluating Binary Classifier Predictions
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
When it comes to evaluating a Binary Classifier, **Accuracy** is a well-known performance metric that is used to tell a strong classification model from one that is weak. Accuracy is, simply put, the total proportion of observations that have been correctly predicted. There are four (4) main components that comprise the mathematical formula for calculating Accuracy, viz. **TP**, **TN**, **FP**, **FN**, and these components grant us the ability to explore other ML Model Evaluation Metrics. The formula for calculating accuracy is as follows:

#### The formula: 
<img src="https://latex.codecogs.com/svg.image?\large&space;\mathbf{{\color{Purple}&space;Accuracy=&space;\frac{TP&plus;TN}{TP&plus;FP&plus;TN&plus;FN}&space;}" title="https://latex.codecogs.com/svg.image?\large \mathbf{{\color{Purple} Accuracy= \frac{TP+TN}{TP+FP+TN+FN} }" align="center"/>

### Precision Score
This refers to the proportion (total number) of all observations that have been predicted to belong to the positive class and are actually positive. The formula for Precision Evaluation Metric is as follows:

<img src="https://latex.codecogs.com/svg.image?\large&space;\mathbf{{\color{Purple}&space;Precision=&space;\frac{TP}{TP&plus;FP}&space;}" title="https://latex.codecogs.com/svg.image?\large \mathbf{{\color{Purple} Precision= \frac{TP}{TP+FP} }" align="center" />












## Model Performance Evaluation - Regression
