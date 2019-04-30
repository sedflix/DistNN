#include "layers.h"
#include <cuda_runtime.h>
#include <math.h>
#include <thrust/extrema.h>
// Layer::Layer(int input_dim, int output_dim)
// {

// }

__global__ void relu_backward(float *input_grad, float *cache, float *output_grad, int N)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(idx < N) {
        output_grad[idx] = input_grad[idx];
        if(cache[idx] < 0) {
            output_grad[idx] = 0;
        } 
    }
}

__global__ void relu_forward(float *x, int N)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx < N) {
        if(x[idx] < 0) {
            x[idx] = 0;
        }
    }
}

__global__ void softmax_backward(float *x, float *real, int N)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx < N)
    {
        x[idx] -= real[idx];
    }
}

__global__ void add(float *a, float *b, int N)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx < N) {
       a[idx] += b[idx];
    }
}

__global__ void subtract(float *a, float *b, int N)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx < N) {
       a[idx] -= b[idx];
    }
}


__global__ void softmax(float *x, int N, int max)
{   
    // float max = *thrust::max_element(x,x+N);
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
}