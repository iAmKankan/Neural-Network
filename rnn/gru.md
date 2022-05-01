## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

## Gated Recurrent Unit (GRU)
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
There are many other variants of the LSTM cell. One particularly popular variant is the GRU cell. The **Gated Recurrent Unit (GRU)** cell was proposed by Kyunghyun Cho et al. in a 2014 paper that also introduced the **Encoder–Decoder network**.
<img src="https://user-images.githubusercontent.com/12748752/166147387-cf5a0e60-78a1-4355-88b3-4e3b2f2e3ad8.png" width=60%/>

The main simplifications of GRU:
* Both state _vectors_ are merged into a single vector **h<sub>(t)</sub>** . 
* A single gate controller **z<sub>(t)</sub>** controls both the **_forget gate_** and the **_input gate_**. If the gate controller outputs a **1**, the forget gate is open (**= 1**) and the input gate is closed (**1 – 1 = 0**). If it outputs a **0**, the opposite happens. In other words, whenever a memory must be stored, the location where it will be stored is erased first. This is actually a frequent variant to the LSTM cell in and of itself. 
* There is _no_ **output gate**; the full state vector is output at every time step. However, there is a new gate controller **r<sub>(t)</sub>** that controls which part of the previous state will be shown to the main layer (**g<sub>(t)</sub>** ).

<img src="https://latex.codecogs.com/svg.image?\\\mathbf{g_{(t)}&space;=&space;tanh(W_{xg}^\top&space;x_{(t)}&plus;W_{hg}^\top&space;(r_{(t)}&space;\otimes&space;h_{(t-1)})&plus;b_g&space;)}\\&space;\\\mathbf{z_{(t)}&space;=&space;\sigma(W_{xz}^\top&space;x_{(t)}&plus;W_{hz}^\top&space;x_{(t-1)}&plus;b_z&space;)}&space;\\&space;\\\mathbf{r_{(t)}&space;=&space;\sigma(W_{xr}^\top&space;x_{(t)}&plus;W_{hr}^\top&space;x_{(t-1)}&plus;b_r&space;)}\\&space;\\\mathbf{h_{(t)}&space;=&space;z_{(t)}&space;\otimes&space;h_{(t-1)}&space;&plus;&space;(1-z_{(t)})\otimes&space;g_{(t)}" title="https://latex.codecogs.com/svg.image?\\\mathbf{g_{(t)} = tanh(W_{xg}^\top x_{(t)}+W_{hg}^\top (r_{(t)} \otimes h_{(t-1)})+b_g )}\\ \\\mathbf{z_{(t)} = \sigma(W_{xz}^\top x_{(t)}+W_{hz}^\top x_{(t-1)}+b_z )} \\ \\\mathbf{r_{(t)} = \sigma(W_{xr}^\top x_{(t)}+W_{hr}^\top x_{(t-1)}+b_r )}\\ \\\mathbf{h_{(t)} = z_{(t)} \otimes h_{(t-1)} + (1-z_{(t)})\otimes g_{(t)}" />

## Bibliography
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition by Aurélien Géron**
