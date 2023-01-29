## Index 
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

## Tanh
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)

* Output in the range -1 to 1
* Saturation problems exists
* Acts like identity near origin- Sigmoid is also Linear near origin but not identity
### Properties:
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
$$ \large \begin{matrix}
& {\color{Blue} \textbf{Domain(input range)}\ \ \tanh(x)} & {\color{Blue} \textbf{Range(output range)} \ \ \tanh(x)} & {\color{Blue} \textbf{Thresold value} \ \ \tanh(x)} & {\color{Blue} \textbf{Derivative} \ \ \frac{\partial }{\partial x} \tanh(x)} \\
& {\color{DarkRed} (\mathbf{-\infty,-1})} & {\color{DarkRed}\textbf{(-1,1)} } & {\color{DarkRed} \textbf{0}} & {\color{DarkRed} \textbf{1}} \\ \\
\end{matrix}
$$ 

### Formula: 
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

* $\Large{\color{Purple} tanh(z) = \frac{sinh(z)}{cos(z)} }$ . Where we have $\Large{\color{Purple} sinh(z) = \frac{e_z - e^{-z}}{2} }\normalsize \cdots \cdots \textrm{(1)}$ and $\Large{\color{Purple} cosh(z) = \frac{e_z + e^{-z}}{2} } \normalsize \cdots \cdots \textrm{(2)}$
* By combining  **1** and **2** we get -

$$\huge{\color{Purple} tanh(z) = \frac{e_z - e^{-z}}{e_z + e^{-z}} }$$

### The Curve
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

<p align="center">
<img src="https://user-images.githubusercontent.com/12748752/187023656-dcfef30c-5923-4a9b-840e-3480cd9f3638.png" width=30%/> 
<br><ins><b><i>Tanh function</i></b></ins>
</p>
 
### Pros
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* The gradient is stronger for tanh than sigmoid ( derivatives are steeper).
### Cons
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* Tanh also has the vanishing gradient problem.
