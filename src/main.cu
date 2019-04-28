// #define USE_MNIST_LOADER
// #define MNIST_DOUBLE
//#include "utils/mnist.h"
#include "layers/layers.h"
#include <stdio.h>


int main() {

    // mnist_data *data;
    // unsigned int cnt;
    // int ret;

    // if (ret = mnist_load("train-images-idx3-ubyte", "train-labels-idx1-ubyte", &data, &cnt)) 
    // {
    //             printf("An error occured: %d\n", ret); 
    //             return 1;
    // } 
    
    // printf("image count: %d\n", cnt);

    
    float *data = (float *)malloc(10*sizeof(float));
    for(int i=0; i<10; i++) {
        data[i] = 1.0f;
    }
    Matrix<float> x = Matrix<float>(10,1,1,data);
    x.to_gpu();

    float *loss = (float *)malloc(3*sizeof(float));
    for(int i=0; i<3; i++) {
        data[i] = 1.0f;
    }
    Matrix<float> loss_m = Matrix<float>(3,1,1,loss);
    loss_m.to_gpu();

    Layer a = Layer(10,3);
    a.forward(x);
    a.backward(loss_m);
    a.update();
    a.update();
    a.update();
    
    cudaDeviceSynchronize();
    a.x.to_cpu();
    a.x.print();

}
