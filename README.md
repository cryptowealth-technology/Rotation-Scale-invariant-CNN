# Rotation-Scale-invariant-CNN
Convolutional neural network (2D index) approximate invariant properties implicitly by max-pooling. Thus filters fail to generalize with drastic changes in scale and rotation. In this page, I show 2D convolution on "Frequency" &amp; "Direction" is a natural way to think of scaling and rotation.

The method proposed here can handle images with different sizes.


# Background
Fourier transform is a decomposition method that can decompose any function into a series of characteristic functions of the heat equation.
In 2D, chararcteristic functions are wave-like, each wave has two parameters **Frequency** and **Direction**.



# Image Preprossing
Instead of using $cos(k\pi), sin(k\pi)$, geometric decay series was used here, explain below.



# Challenge 
If only part of the image shown in the picture, signal maynot strong enough for filters. 
Need some localization pattern detecting, Which is time domain convolution (traditional one), to detect wave-like pattern.
