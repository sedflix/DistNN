// #define USE_MNIST_LOADER
// #define MNIST_DOUBLE
//#include "utils/mnist.h"
#include "layers/layers.h"
#include <stdio.h>
#include <algorithm>
#include <math.h>
#include <thrust/extrema.h>


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
    float label[] = {0,1,0};
    for(int i=0; i<10; i++) {
        data[i] = 1.0f;
    }
    Matrix<float> x = Matrix<float>(10,1,1,data);
    x.to_gpu();

    int input_dimension = 10;
    int num_hidden = 2;
    int hidden_sizes[] = {5,4};
    int num_classes = 3;
    Network net = Network(input_dimension,num_classes,num_hidden, hidden_sizes);
    
    Matrix<float> output = Matrix<float>(num_classes,1,1);
    
    output = net.forward(x);
    // output.to_cpu();
    // output.print();

    thrust::device_ptr<float> d_ptr = thrust::device_pointer_cast(output.get_d());
    float max = *(thrust::max_element(d_ptr, d_ptr + num_classes));
    softmax<<<1,num_classes>>>(output.get_d(), num_classes, max);
    // output.to_cpu();
    // output.print();

    float *predicted = output.get_d();
    float *real_label;
    cudaMalloc((void **)&real_label, num_classes * sizeof(float));
    cudaMemcpy(real_label,label, num_classes * sizeof(float), cudaMemcpyHostToDevice);
    softmax_backward<<<num_classes,1>>>(predicted, real_label, num_classes);
    output.to_cpu();
    output.print();
    // Matrix<float> loss = Matrix<float>(num_classes,1,1,predicted);

    // output.to_cpu();
    // output.print();
    // temp = softmax(temp,num_classes);

    // Matrix<float> loss = Matrix<float>(num_classes,1,1,temp);
    // net.backward(loss);

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

// float *softmax(float *x, int N)
// {   
//     float max = *std::max_element(x,x+N);
//     float sum = 0;
//     for (int i = 0; i<N; i++)
//     {
//         x[i] -= max;
//         x[i] = exp(x[i]);
//         sum += x[i];
//     }
//     for (int i = 0; i<N; i++)
//     {
//         x[i]/=sum;
//     }
//     return x;
// }