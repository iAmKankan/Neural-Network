## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
* [Activation Function](#activation-function)
   * [Sigmoid](https://github.com/iAmKankan/Neural-Network/blob/main/activation_functions/sigmoid.md)
   * [Softmax](https://github.com/iAmKankan/Neural-Network/blob/main/activation_functions/softmax.md)
   * [TanH](#tanh)






## Activation Function
![dark](https://user-images.githubusercontent.com/12748752/136802585-2ef5b7ff-ddbc-417f-b963-ca233db3ded1.png)

### Why do we need activation functions in the first place
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
* If you chain several linear transformations, all you get is a linear transformation.
> **For example: Say we have f(x) and g(x) then Then chaining these two linear functions gives you another linear function f(g(x)).**
>> f(x) = 2 x + 3 
>
>> g(x) = 5 x - 1 
>
>> f(g(x)) = 2(5 x - 1) + 3 = 10 x + 1.


* So, if you don’t have some non-linearity between layers, then even a deep stack of layers is equivalent to a single layer.
* You cannot solve very complex problems with that.

> ### The botton line is _linear activation function_ cannot be used in _hidden layers_, it has to be at the end if there is a requirment i.e for _regression output layer_ for some special cases
