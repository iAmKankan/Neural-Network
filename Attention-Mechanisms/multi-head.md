## Index:
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

## Multi-Head Attention
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

In practice, given the same set of **queries**, **keys**, and **values** we may want our model to combine knowledge from different behaviors of the same attention mechanism, such as capturing dependencies of various ranges (e.g., **shorter-range** vs. **longer-range**) within a sequence. Thus, it may be beneficial to allow our attention mechanism to jointly use different representation subspaces of **queries, keys, and values**.

To this end, instead of performing a single attention pooling, queries, keys, and values can be transformed with  independently learned linear projections. Then these  projected queries, keys, and values are fed into **attention pooling in parallel**. 

In the end,  attention pooling outputs are concatenated and transformed with another learned linear projection to produce the final output. 

This design is called multi-head attention, where each of the  attention pooling outputs is a head. Using **fully-connected layers** to perform **learnable linear transformations**.

<img src="https://user-images.githubusercontent.com/12748752/170055315-b69b2b13-f3a5-44c6-8a6a-6a4655359f80.png" width=60%/>
<ins><i><b>Multi-head attention, where multiple heads are concatenated then linearly transformed.</b></i></ins>
