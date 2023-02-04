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
One of the normalization techniques that can be that, all the attribute values can vary from say **0** to **1**. So, I will normalize all the attribute values in such a way that no attribute will have a value less than **0** and no other attribute will have a value greater than **1**. So, that the contribution of all those attributes in the **final decision making process** is more or less same or all the **attributes** are **equally weighted**.

* $\Large{\color{Purple}\mathrm{x}}$ is the population of different instances of attributes. Say for example, I have different customers or different individuals having ages ranging from say **18** to **100** and I may have say 100 such customers. 

#### What is the average age?
* So, the average age is nothing but $\Large{\color{Purple} \mathrm{\mu_x} = \mathrm{\frac{1}{N} \Sigma_{i=1}^{100} x_i} }$, where $\Large{\color{Purple}\mathrm{i}}$ will vary from **1** to **100**, if I have **100** such customers.

$$\Huge{\color{Purple} 
\Huge\mathrm{\hat{x}} = \frac{\mathrm{x - \mu_x} }{\sigma}
}
$$

* Where $\Large{\color{Purple}\mathrm{\mu_x}}$ is the **mean** of these attributes and **standard deviation** $\Large{\color{Purple}\mathrm{\sigma}}$ , I compute what is the standard deviation of all these attribute values that I have or all these instances that I have. So, I can normalize it with respect to the standard deviation of the all the attributes. So, that is what becomes $\Large{\color{Purple}\mathrm{x}}$ normalized or$\Large{\color{Purple}\mathrm{\hat{x}}}$ .
