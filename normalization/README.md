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

$$\Huge{\color{Purple}\begin{matrix}
\mathrm{\mu_x} = \mathrm{\frac{1}{N} \Sigma_{i=1}^{100} x_i} &
\left\{\begin{matrix}
\large \textbf{x} \textrm{ is the poulation of different instances of attributes.} \\
\end{matrix}\right.
\end{matrix}
$$

