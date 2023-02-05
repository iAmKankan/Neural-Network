## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

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




## References:
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* [NPTEL Deep Learning, Prof. Prabir Kumar Biswas IITKGP](https://onlinecourses.nptel.ac.in/noc21_cs05/course)
