## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

> #### _Sigmoid Function_
<img src="https://latex.codecogs.com/svg.image?\frac{1}{1&plus;e^{{-x}}}" title="\frac{1}{1+e^{{-x}}}" width=8% />

> #### _Derivatives Of Sigmoid Function_
* <img src="https://latex.codecogs.com/svg.image?\frac{d}{dx}f=\frac{(denominator&space;*&space;\frac{d}{dx}numerator)&space;-&space;(numerator&space;*&space;\frac{d}{dx}denominator)}{denominator^2}" title="\frac{d}{dx}f=\frac{(denominator * \frac{d}{dx}numerator) - (numerator * \frac{d}{dx}denominator)}{denominator^2}" />
* <img src="https://latex.codecogs.com/svg.image?\frac{d}{dx}S(x)=\frac{1}{1&plus;e^{{-x}}}" title="\frac{d}{dx}S(x)=\frac{1}{1+e^{{-x}}}" />
* <img src="https://latex.codecogs.com/svg.image?\frac{d}{dx}S(x)=\frac{((1&plus;e^{-x})&space;*&space;\frac{d}{dx}1)&space;-&space;(1&space;*&space;\frac{d}{dx}(1&plus;e^{-x}))}{({1&plus;e^{-x}})^2}" title="\frac{d}{dx}S(x)=\frac{((1+e^{-x}) * \frac{d}{dx}1) - (1 * \frac{d}{dx}(1+e^{-x}))}{({1+e^{-x}})^2}" />
* <img src="https://latex.codecogs.com/svg.image?\frac{d}{dx}S(x)=\frac{({1&plus;e^{-x}})&space;(0)&space;-&space;(1)&space;(-e^{-x})}{({1&plus;e^{-x}})^2}&space;" title="\frac{d}{dx}S(x)=\frac{({1+e^{-x}}) (0) - (1) (-e^{-x})}{({1+e^{-x}})^2} " />
* <img src="https://latex.codecogs.com/svg.image?\frac{d}{dx}S(x)=\frac{e^{-x}}{({1&plus;e^{-x}})^2}&space;" title="\frac{d}{dx}S(x)=\frac{e^{-x}}{({1+e^{-x}})^2} " />
* <img src="https://latex.codecogs.com/svg.image?\frac{d}{dx}S(x)=\frac{1-1&plus;e^{-x}}{({1&plus;e^{-x}})^2}&space;" title="\frac{d}{dx}S(x)=\frac{1-1+e^{-x}}{({1+e^{-x}})^2} " />
* <img src="https://latex.codecogs.com/svg.image?\frac{d}{dx}S(x)=\frac{1&plus;e^{-x}}{({1&plus;e^{-x}})^2}-\frac{1}{(1&plus;e^{-x})^2}" title="\frac{d}{dx}S(x)=\frac{1+e^{-x}}{({1+e^{-x}})^2}-\frac{1}{(1+e^{-x})^2}" />
* <img src="https://latex.codecogs.com/svg.image?\frac{d}{dx}S(x)=\frac{1}{({1&plus;e^{-x}})}-\frac{1}{(1&plus;e^{-x})^2}" title="\frac{d}{dx}S(x)=\frac{1}{({1+e^{-x}})}-\frac{1}{(1+e^{-x})^2}" />
* <img src="https://latex.codecogs.com/svg.image?\frac{d}{dx}S(x)=\frac{1}{({1&plus;e^{-x}})}\left&space;(&space;1-\frac{1}{(1&plus;e^{-x})}&space;\right&space;)" title="\frac{d}{dx}S(x)=\frac{1}{({1+e^{-x}})}\left ( 1-\frac{1}{(1+e^{-x})} \right )" />
