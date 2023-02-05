## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

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

I compute what is the **standard deviation** of _all these attribute values that I have_ or _all these instances that I have_. So, I can **normalize** it with respect to the **standard deviation** of the all the attributes. So, that is what becomes $\Large{\color{Purple}\mathrm{x}}$ **normalized** or $\Large{\color{Purple}\mathrm{\hat{x}}}$ .



$\Large{\color{Purple}\mathrm{\hat{x}}}$ will have a a mean value or $\Large{\color{Purple}\mathrm{\mu}}$ which is equal to **0** ( $\Large{\color{Purple}\mathrm{\mu = 0}}$ )and it will have an standard deviation say $\Large{\color{Purple}\mathrm{\sigma_{\hat{x}}}}$ which will be equal to **1** ( $\Large{\color{Purple}\mathrm{\sigma_{\hat{x}} = 1}}$ )because you are normalizing with respect to **standard deviation**. 

So, this is one form of **normalization** where you are making the **mean** of the attributes which will be equal to **0** and the **variance** is **1** **_because variance is nothing but square of the standard deviation_**. So, **variance** or **standard deviation** will be equal to **1** and this will be done for all the attributes. So, even the attribute of **age** will have a **mean 0** and **standard deviation 1** and the attributes which are **income** the different instances of income that will also have **mean 0** and **standard deviation 1**. So, this is one form of **normalization** that can be done.


### Min-Max
This normalization the attribute values will be **eather 0 or 1**. So, the minimum attribute value will be **0** and the maximum attribute value will be **1**.

So, this is one form of normalization where you are making the mean of the attributes which will be equal to 0 and the variance is 1 because variance is nothing but square of the standard deviation. So, variance or standard deviation will be equal to 1 and this will be done for all the attributes. So, even the attribute of age will have a mean 0 and standard deviation 1 and the attributes which are income the different instances of income that will also have mean 0 and standard deviation 1. So, this is one form of normalization that can be done.

$$\Huge{\color{Purple} 
\mathrm{\hat{x}} = \frac{\mathrm{x - x_{mean}} }{\mathrm{x_{max}-x_{mean}}}
}
$$



### What does a classifier learn?
