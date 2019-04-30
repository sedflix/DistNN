// #define USE_MNIST_LOADER
// #define MNIST_DOUBLE
//#include "utils/mnist.h"
#include "layers/layers.h"
#include <stdio.h>
#include <algorithm>
#include <math.h>

float *softmax(float *,int);

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
    float *label;
    for(int i=0; i<10; i++) {
        data[i] = 1.0f;
    }
    Matrix<float> x = Matrix<float>(10,1,1,data);
    x.to_gpu();

    int input_dimension = 10;
    int num_hidden = 2;
    int hidden_sizes[] = {5,4};
    int num_classes = 2;
    Network net = Network(input_dimension,num_classes,num_hidden, hidden_sizes);
    
    Matrix<float> output = Matrix<float>(num_classes,1,1);
    
    output = net.forward(x);
    output.to_cpu();

    float *temp = (float *)malloc(num_classes * sizeof(float));
    temp = output.get_h();
    temp = softmax(temp,num_classes);

    float *der = cudaMalloc(num_classes * sizeof(float));
    cudaMemcpy(der,temp,num_classes * sizeof(float),cudaMemcpyHostToDevice);
    softmax_backward<<<num_classes,1>>>(der,label,num_classes);
    cudaMemcpy(temp,der,num_classes * sizeof(float),cudaMemCpyDeviceToHost);
    Matrix<float> loss = Matrix<float>(num_classes,1,1,temp);
    net.backpropagation(loss);

    // float *loss = (float *)malloc(3*sizeof(float));
    // for(int i=0; i<3; i++) {
    //     data[i] = 1.0f;
    // }
    // Matrix<float> loss_m = Matrix<float>(3,1,1,loss);
    // loss_m.to_gpu();

    // Layer a = Layer(10,3);
    // printf("what\n");
    // a.forward(x);
    // a.backward(loss_m);
    // a.update();
    // a.update();
    // a.update();
    
    cudaDeviceSynchronize();
    // a.x.to_cpu();
    // a.x.print();

}

float *softmax(float *x, int N)
{   
    float max = *std::max_element(x,x+N);
    float sum = 0;
    for (int i = 0; i<N; i++)
    {
        x[i] -= max;
        x[i] = exp(x[i]);
        sum += x[i];
    }
    for (int i = 0; i<N; i++)
    {
        x[i]/=sum;
    }
    return x;
}