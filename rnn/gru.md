## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

### Recap: Vanilla RNN or Plain RNN

<p align="center">
 
 <img src="https://user-images.githubusercontent.com/12748752/188993131-2226bcb8-568f-4c6e-bf62-b23e602ef9d5.png" width=50%/>
 <br><ins><b><i>Vanilla RNN</i></b></ins> 
 
</p>
 
 * So the **input layer** is **x<sub>t</sub>** and there is also **another input coming from the previous layer** which is **h<sub>t-1</sub>** where **_t_** is the level at which we are looking and the output or the input to the next layer is  **h<sub>t</sub>** and the output here is **y<sub>t</sub>**. 
 * The formulation is simply a **linear combination** in this case **W** and **U** are **weight matrices**,  this linear combination followed by **an non-linearity**, the typically in **RNN** is a **tanh** layer.
 *  Now the output which is optional you can take out the output at any point.
 *  **Vanilla RNNs** have trouble with either **vanishing** or **exploring gradients** during **back propagation** - because repeated operation of this sort can actually make the gradients **either increase continuously** or **decrease continuously** depending on the **eigenvalues** of these **matrices**.
 
 > #### The Basic Idea: Need some kind of memory

### Simplified GRU (Andru NG)- (Not a practiced Algo)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
Let us come back to our original vanilla RNN architecture we had **x<sub>t</sub>**, we have **h<sub>t-1</sub>**, **h<sub>t</sub>** comes out so we say in general that this is the output of the **vanilla RNN**.

$$ \Huge{\color{Purple} h_t = tanh (W_{hh}h_{t-1}+W_{xh} x_t)} $$
 
Simplified GRU labels this instead of calling the followings-
   * **h<sub>t</sub>** as **g**,
   *  **linear combination** is always called **z**, so we will call it **g= tanh(z<sub>g</sub>)**, where **[z<sub>g</sub>** = **W h<sub>t-1</sub> + Ux<sub>t</sub>]**, 
   *  **W** and **U** are matrices and for a particular reason we will called them **W<sub>g</sub>** and **U<sub>g</sub>** instead- like this **[W<sub>g</sub> h<sub>t-1</sub> + U<sub>g</sub> x<sub>t</sub>]**
 
#### So far same Vanilla RNN looks like-
 $$ \Huge{\color{Purple} g = tanh (z_g)} $$

#### Simplified GRU
Now if we **combine**(add) **linear combination of "Vanilla RNN"** (**g**) with **previous time stamp output** (previous computation **h<sub>t</sub>**) we get

$$ \Huge{\color{Purple}
\begin{align*}
& h_t = (1-\lambda ) g + \lambda h_{t-1} & \begin{cases}\large \mathrm{\lambda \in [0, 1];} \\ 
\large \mathrm{\lambda \ is\ a\ scaler} \end{cases} \\
\end{align*}
} 
$$

* When  **&lambda;** = 1 &rArr; **h<sub>t</sub>** = **h<sub>t-1</sub>** &rArr; **Pure Momory**
* When  **&lambda;** = 0 &rArr; **h<sub>t</sub>** = **g** &rarr; **"Vanilla RNN"** 

#### The final equation 

$$ \Huge{\color{Purple}
\begin{align*}
& h_t = (1- f) \odot g  + f \odot h_{t-1}  &  
\end{align*}
\normalsize \begin{cases}
f \in \[0,1\] \\
[f \odot h_{t-1}] \Rightarrow f \textrm{ is a Forget Gate} \\
When,\ f \textrm{ = 1 become a Memory Cell ; } When,\ f \textrm{ = 0 become a Vanilla RNN} \\
\end{cases}
} 
$$ 

### Difference between the followings 

$$\begin{align*}
& \large {\color{Purple} (1- \lambda)g} & \textrm{ vs } & & \large{\color{Purple} (1- f)\odot g} 
\end{align*}
$$

* **&lambda;** is a scaler in Vanilla RNN
* **f** is a vector in simple GRU

#### Why would we take **f** to be a vector?
**Answer:** Because it is possible that you might want to **remember** a few thing and **forget** a things within a vector, within the **h** vector.

#### What value of **f** do we choose?
**Answer:**  Here we use the general principle of whatever we have been doing in neural networks which is we never really specify any component we let the algorithm choose it.

#### The value of _f_ also has to be between 0 and 1 why?
**Answer:** **_f_ &in; [0,1]** Because that is only then does the **linear combination** work out well there only then does it look like interpolation.

Now **_f_ &in; [0,1]** is same as **&sigmoid;** so- I want f to be between 0 and 1, you remember from logistic re- gression that we have one function that always squeezes, any function into 0 and 1, so the same principle apply here, so we will kee f as a **&sigmoid;** of something, a **&sigmoid;** of what, same we have linear combination of only two vector here at any place you have these two vectors **h<sub>t</sub>**, **h<sub>t-1</sub>** and **x<sub>t</sub>** and you make them a linear combination of this.

$${\color{Purple}
\large f = \sigma(W_f h_{t-1} + U_f x_t)
}
$$

### Summary

$$\Huge{\color{Purple}
\begin{align*}
& h_t = f \odot h_{1-t}  + (1 - f) \odot g \\
& g = tanh (z_g); & \mathrm{z_g = W_g h_{t - 1} U_g x_t} \\
& f = \sigma (z_f); & \mathrm{z_f = W_f h_{t - 1} U_f x_t} \\
\end{align*}
}
$$

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
