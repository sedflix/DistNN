# DistNN

##  Usage

```
cd src
make
./main
```

## Code Description

**/layers/layers.cu/h** : This file implements a basic abstraction of a layer in a neural network. It has a class Layer which is responsible for storing x,y,b and their corresponding derivatives. It also implements the relu activation, backprop, addtion kernel and other operations that are usually performed on a layer.

**/utils/matrix.cu/h** : This file implements a normal matrix class to ease the allocation of GPU memory, matrix multiplication and moving of data between cpu and gpu.