## Index:
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)


## ⬛ Self-Attention and Positional Encoding
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
In order to encode a sequence in deep learning, we often use **CNNs** or **RNNs**.

**Now**, with _attention mechanisms_, imagine that we feed **_a sequence of tokens_** into **_attention pooling_** so that the same set of tokens act as **queries**, **keys**, and **values**. Specifically, each **query** attends to all the **_key-value pairs_** and generates one attention output. 

Since the queries, keys, and values come from the same place, this performs **_self-attention_**, which is also called **_intra-attention_**.

Lets discuss sequence encoding using self-attention, including using additional information for the sequence order.

### 🔲Self-Attention
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
