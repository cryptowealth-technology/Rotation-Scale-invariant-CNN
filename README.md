# Rotation-Scale-invariant-CNN
Convolutional neural network (2D index) approximate invariant properties implicitly by max-pooling. Thus filters fail to generalize with drastic changes in scale and rotation. In this page, I show 2D convolution on **`Frequency`** &amp; **`Direction`** is a natural way to think of scale and rotation.

# Pros & Cons

* Pros:
  1. Images size are normalized at preprossing stage, filters are now capturing patterns between images of different sizes.
  2. GlobalMaxPooling makes convoution simple, you no longer need to stack deep convolutional networks.

* Cons:
  1. Fourier series are not `correctly` used here, not sure about the theoretical background of geometric frequency series.
  2. Propotional area of the desired pattern in the image plays a important role. Hope nonlinear activation can help dealing with it.

# Background
Fourier transform is a decomposition method that can decompose any function into weighted sum of series of characteristic functions of the heat equation.
In 2D, chararcteristic functions are wave-like, each wave has two parameters **`Frequency`** and **`Direction`**.

<img src="icon/characteristic.png" width=600>

# Image Preprossing
Instead of using ![](https://latex.codecogs.com/svg.latex?cos(k\omega),\;\;%20k\in%20N),
geometric decay series was used here for scaling property.

In preprossing stage, images were decompose into linear combination of series of waves with differ frequencies and directions. 

Notice: ![](https://latex.codecogs.com/svg.latex?a%20\cdot%20cos(w)%20+%20b%20\cdot%20sin(w)%20=%20\sqrt{a^2+b^2}%20\cdot%20cos(w%27)=c%20\cdot%20cos(\omega%27))

<img src="icon/padding.png" width=600>

# Invariant Properties

If we apply 2D convolution on these feature map, with `kernel size = (n, d)`. #`d` had better equal to number of direction

We can see that shift of filter in **`Direction`** is rotation of a image, and **`frequency`** would mean scale. Since we choose a small decay rate, different scale are considered.

<img src="icon/conv.png" width=600>

# Stacked filter 
Since composition of convolution operator is another convolution(another filter on original space), of course you can stack multiple filters.

# Challenge 
If only part of the image shown in the picture, signal maynot strong enough for filters. 
Need some localization pattern detecting, Which is time domain convolution (traditional one), to detect wave-like pattern.
