## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

### Example #1
Suppose we have a dataset of **Bank a/c holders** and we have taken two features from it **Age** and other is **Income**. Now lets see those features- 

$$\Large{\color{Purple}
\begin{matrix}
\\ {\textbf{Label}} & {\textbf{Numeric Ranges}} & {\textbf{Difference}} \\ 
\hline
Age                  & \textrm{18 years to 100 years } &  \textbf{82}\\ 
\hline
Income               &  \textrm{10,000 to 10,00000}  &  \textbf{9,90,000} \\ 
\hline
\end{matrix}
}
$$


So, considering just these two attributes, you will find that **Age** attribute has **a very small range**, **very narrow range** from **18 years** to **100 years**. Whereas, the other attribute which is monthly **Income** that has a **wide range** vary from **10,000 rupees a month** to even **10 lakh rupees a month** . 
#### What is the tenure of loan that can be sanctioned to an individua based on these two attributes? 

The **decission will be biased** by the attribute which is monthly **Income** because this is **many times larger** than the other attribute which is **Age** or the **range is very large compared to the values and range of other attributes**. 

So, in order to make your decision **unbiased** you need some sort of **normalization** techniques.

### Z-Score
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

In this **normalization technique**, all the **attribute values can vary** from **0** to **1**. So, that the contribution of all those attributes in the **final decision making process** is more or less same or all the **attributes** are **equally weighted**.

> Suppose, different customers having **ages** ranging from say **18** to **100** and I may have say **100 such customers**. 

#### What is the average age?
* So, the **average** age is nothing but $\Huge{\color{Purple} \mathrm{\mu_x} = \mathrm{\frac{1}{N} \Sigma_{i=1}^{100} x_i} }$,
   * $\Large{\color{Purple}\mathrm{x}}$ is the population of different instances of attributes.
   * Where $\Large{\color{Purple}\mathrm{i}}$ will vary from **1** to **100**, if I have **100** such customers.

$$\Huge{\color{Purple} 
\mathrm{\hat{x}} = \frac{\mathrm{x - \mu_x} }{\sigma}
}
$$

Where as
* $\Huge{\color{Purple}\mathrm{\mu_x} = }\normalsize{\color{Cyan}\textbf{ Mean}}$ 
* $\Huge{\color{Purple}\mathrm{\sigma}= }\normalsize{\color{Cyan}\textbf{ Standard Deviation}}$ ,

I compute what is the [**standard deviation**](https://github.com/iAmKankan/Statistics/blob/main/measureofcentraltendency.md#standard-deviation) of _all these attribute values that I have_ or _all these instances that I have_. So, I can **normalize** it with respect to the [**standard deviation**](https://github.com/iAmKankan/Statistics/blob/main/measureofcentraltendency.md#standard-deviation) of the all the attributes. So, that is what becomes $\Large{\color{Purple}\mathrm{x}}$ **normalized** or $\Large{\color{Purple}\mathrm{\hat{x}}}$ .



$\Large{\color{Purple}\mathrm{\hat{x}}}$ will have a a mean value or $\Large{\color{Purple}\mathrm{\mu}}$ which is equal to **0** ( $\Large{\color{Purple}\mathrm{\mu = 0}}$ )and it will have an standard deviation say $\Large{\color{Purple}\mathrm{\sigma_{\hat{x}}}}$ which will be equal to **1** ( $\Large{\color{Purple}\mathrm{\sigma_{\hat{x}} = 1}}$ )because you are normalizing with respect to **standard deviation**. 

So, this is one form of **normalization** where you are making the **mean** of the attributes which will be equal to **0** and the [**variance**](https://github.com/iAmKankan/Statistics/blob/main/measureofcentraltendency.md#varience) is **1** **_because variance is nothing but square of the standard deviation_**. So, [**variance**](https://github.com/iAmKankan/Statistics/blob/main/measureofcentraltendency.md#varience) or [**standard deviation**](https://github.com/iAmKankan/Statistics/blob/main/measureofcentraltendency.md#standard-deviation) will be equal to **1** and this will be done for all the attributes. So, even the attribute of **age** will have a **mean 0** and **standard deviation 1** and the attributes which are **income** the different instances of income that will also have **mean 0** and **standard deviation 1**. So, this is one form of **normalization** that can be done.


### Min-Max
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

This normalization the attribute values will be **eather 0 or 1**. So, the minimum attribute value will be **0** and the maximum attribute value will be **1**.

So, this is one form of normalization where you are making the mean of the attributes which will be equal to 0 and the variance is 1 because variance is nothing but square of the standard deviation. So, variance or standard deviation will be equal to 1 and this will be done for all the attributes. So, even the attribute of age will have a mean 0 and standard deviation 1 and the attributes which are income the different instances of income that will also have mean 0 and standard deviation 1. So, this is one form of normalization that can be done.

$$\Huge{\color{Purple} 
\mathrm{\hat{x}} = \frac{\mathrm{x - x_{mean}} }{\mathrm{x_{max}-x_{mean}}}
}
$$



### What does a classifier learn?
For binary classification


$$\Huge{\color{Purple} 
\mathrm{P(Y/X)} 
\ \ \ \ \ \normalsize \left \\{ \begin{matrix}\textrm{ X = input data} \\
\textrm{Y = class index} 
\end{matrix} \right.
}
$$

you give input as appear say $\Large{\color{Purple}\mathrm{(X_i, Y_i)}}$ which indicates that this data  $\Large{\color{Purple}\mathrm{X_i}}$ belongs to class $\Large{\color{Purple}\mathrm{Y_i}}$ So these are the labelled data which are fed for training of the classifier or training of the deep neural network.

## <ins><i>Covariate Shift</i></ins> on Batch Processing
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

Training data are fed to the neural network or fed to the classifier during **learning operation** not all in a one short rather in the form of **batches**. And it is quite possible that the distribution of the data in **one batch may be different** from the **distribution of data in another batch** even though the data comes from the **same class**.

<p align="center">
  <img src="https://user-images.githubusercontent.com/12748752/216840751-68dcd43a-3743-487a-b9ba-8d6b4e18d6a1.png" width=40%/>
  <img src="https://user-images.githubusercontent.com/12748752/216840735-7ec5cf2b-516e-4c3b-a6a1-7b035ac88f2f.png" width=40%/>
  <br><ins><b><i>Covariate Shift</i></b></ins>
</p>

So, as a result you find that depending upon the characteristics of the input data or in this case just the colour of the input data the distribution of the features of the images taken from flower category and the distribution of the features taken from images of the nonflower category they are different and as a result the **classifier which is learnt with the first batch** is **different** from the **classifier that is learnt from the second batch**. 

So, this is the problem this is what is known as **covariate shift** and because of this covariate shift now because the **classifier has to hop** from one classifier to another classifier **during learning**, your learning process eventually becomes **very, very slow**.


### Recap
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
As we saw in **Covariate Shift**

<p align="center">
  <img src="https://user-images.githubusercontent.com/12748752/216840751-68dcd43a-3743-487a-b9ba-8d6b4e18d6a1.png" width=40%/>
  <img src="https://user-images.githubusercontent.com/12748752/216840735-7ec5cf2b-516e-4c3b-a6a1-7b035ac88f2f.png" width=40%/>
  <br><ins><b><i>Covariate Shift</i></b></ins>
</p>

So, now we find that though the images belonging to the same category of flowers, but because of their appearance the computed features may have **different distribution**. And as a result **while training the classifier** simply **hops from one boundary to another boundary**. In some cases it will decide this left boundary, in some cases it will decide this right boundary. So, as a result the time taken to train the **classifier** or the **time taken to train** your deep neural network becomes **very large**.


## Solution
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
So, this can be avoided if we can somehow normalize the feature vectors, so that the distribution of **all the feature vectors will belonging to the same class** will be more or less same. And the kind of normalization that we can apply in this case is a [**z-score**](https://github.com/iAmKankan/Neural-Network/edit/main/normalization/README.md#z-score) type of normalization that we have already discussed.

$$\Huge{\color{Purple} 
\mathrm{\hat{x}} = \frac{\mathrm{x - \mu_x} }{\sqrt{\epsilon + \sigma_x ^2}}
}
$$

$${\color{Purple}\begin{matrix}
& X &=& {\color{Cyan}\textrm{ Set of feature vectors} }\\
& \hat{X} &=& {\color{Cyan}\textrm{ Normalised X} }\\
& \mu_x &=& {\color{Cyan}\textrm{ the mean of the feature vectors}} \\
& \sigma_x &=& {\color{Cyan}\textrm{ standard deviation of the feature vectors}} \\
& \epsilon &=& {\color{Cyan}\textrm{ very small value, ensures divide by zero never occurs}} \\
\end{matrix}
} 
$$

## Normalization application layer in Neural Network

This is applicable not only in the **input layer** this is also applicable in the **hidden layers** as well.

## Why should it be applicable to hidden layers? 
So, now, let us try to see that why do we need normalization even in hidden layers. So, for to discuss about that to see why you need normalization in the hidden layers, let us look at a typical architecture of a deep neural network.


<p align="center">
  <img src="https://user-images.githubusercontent.com/12748752/216846018-86d4e1ef-9525-4b93-abf0-81dabf981a2d.png" width=40%/>
</p>



And during back propagation what you do is you compute your error at the output and then following the gradient descent procedure in the backward pass you pass that gradient to the layers from the output side to the input side and while doing so, in every layer you go on updating the weight vectors or updating the parameter vectors. That is what you do. Now, when you come to the updation of or say come to this layer l only, you find that l the layer l gets activations from layer l 1, which are say activations al1 . And based on the distribution of al1 you adjust these weight vectors of layer l . Now, we find that if al1 is steady; that means, in every epoch the distribution of al1 remains the same, then learning of the layer al , will not be a problem because the distribution of al1 which is coming from l 1 layer that remains the same. But what happens? That during this training process this layer al1 is also updating its weights; that means, the weight vectors from layer l  2 to layer l 1 that is also being updated. The weight vectors from layer l 1 to layer l they are also being updated. So, as a result this al1 , the distribution of this may not remain same over the epochs. So, even if you are feeding the same input in the same batch the distribution of al1 the features which are computed at al1 may be different in different epochs. So, leading to the same problem of **covariate shift**.

So, even here, even in the hidden layers or the internal layers I have to take some action so that this covariate shift can be minimized. So, for minimization of this covariate shift as we have seen before that I have to go for normalization of these weight vectors and in most of the cases what is tried is that you remap this vectors in such a way that the mean of all these feature vectors become 0 with a variance or standard deviation equal to 1. So, with this introduction that I need normalization not only at the input layer on the raw data, I also need normalization even at the hidden layer because the features which are computed from different layers even for the same input over different epoch or different training instances, the distribution of those feature vectors may be different which I had to take some measure to erased that shape of the distribution.

## Normalization types
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
 So, this is how the problem of **covariate shift** has to be addressed. So, there are different ways in which this normalization can be done, one of the technique is 
 1. batch normalization. 
 2. layer normalization, 
 3. instance normalization 
 4. group normalization techniques. 
 
 So, these are the different variants of the normalization techniques the main difference is the way you compute the **mean** $\Huge{\color{Purple}\mathrm{\mu_x} = }\normalsize{\color{Cyan}\textbf{ Mean}}$ and **standard deviation** $\Huge{\color{Purple}\mathrm{\sigma}= }\normalsize{\color{Cyan}\textbf{ Standard Deviation}}$.      
 
## References:
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* [NPTEL Deep Learning, Prof. Prabir Kumar Biswas IITKGP](https://onlinecourses.nptel.ac.in/noc21_cs05/course)
