## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
### Bi-Directional RNN or Bi-LSTM
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
Simply this is a **bi- directional RNN** using **LSTM**, _rather than the usual_ **RNN**, **might be deep** or **not deep**.

In **sequence learning**, we assumed that our goal is to model the **_next output_** given, **in the context of a time series** or **in the context of a language model**. In reality the **_sequence can go both backward and forward_** that is not only does the **future depend on the past**, so to speak but the **past also depends on the future**.  

To illustrate the issue, consider the following three tasks of filling in the blank in a text sequence:

$${\color{Purple}
\large\begin{align*}
& \textrm{I am }\underline{\ \ \ \ } .\\
& \textrm{I am }\underline{\ \ \ \ }  \textrm{ hungry} .\\
& \textrm{I am } \underline{\ \ \ \ }  \textrm{ hungry and I can eat half a pig}.\\
\end{align*}
}
$$

* Depending on the amount of information available, we might fill in the blanks with very different words such as “**happy**”, “**not**”, and “**very**”. Clearly the end of the phrase (if available) conveys significant information about which word to pick. 
* A sequence model that is incapable of taking advantage of this will perform poorly on related tasks. 
   * For instance, to do well in **named entity recognition** (e.g., to recognize whether “**Green**” refers to “**Mr. Green**” or to the color) longer-range context is equally vital. 

#### Example #1
Now, what would be an example of that let me give you a very-very simple example though you can think of several thing even in an engineering problems. I will come back to that, so suppose I write something of this sort and you have a optical character recognition tool which basically means this is handwritten and just like we saw with Mnist you want to recognize a handwritten digit.

