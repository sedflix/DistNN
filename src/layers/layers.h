#include "../utils/matrix.h"
#include <list>
#include <cuda_runtime.h>

__global__ void relu_forward(float *x, int N);

__global__ void backward_w(float *input_grad, float *cache, float *output_grad, int N);
__global__ void backward_x(float *input_grad, float *cache, float *output_grad, int N);
__global__ void relu_backward(float *input_grad, float *cache, float *output_grad, int N);
__global__ void subtract(float *a, float *b, int N);
__global__ void add(float *a, float *b, int N);
__global__ void softmax_backward(float*,float*,int);
class Layer
{
  public:
    int input_dim;
    int output_dim;

    Matrix<float> w;
    Matrix<float> b;

    Matrix<float> dw;
    Matrix<float> db;
    Matrix<float> dx;

    Matrix<float> y;
    Matrix<float> dy;

    Matrix<float> x;

    Layer(int input_dim, int output_dim)
    {
        this->input_dim = input_dim;
        this->output_dim = output_dim;

        w = Matrix<float>(output_dim, input_dim, 1, true);
        dw = Matrix<float>(output_dim, input_dim, 1);

        x = Matrix<float>(input_dim, 1, 1);
        dx = Matrix<float>(input_dim, 1, 1);

        y = Matrix<float>(output_dim, 1, 1);
        dy = Matrix<float>(output_dim, 1, 1);

        y_prime = Matrix<float>(output_dim, 1, 1);

        b = Matrix<float>(output_dim, 1, 1);
        db = Matrix<float>(output_dim, 1, 1);
    }

    ~Layer()
    {
    }

    Matrix<float> forward(Matrix<float> x)
    {
        this->x = x;

        // this->x.print();
        // printf("x\n");
        // this->w.print();
        // printf("w\n");

        // this->x.to_gpu();
        // this->w.to_gpu();
        // this->y.to_gpu();
        matrix_mul(&this->w, &this->x, &this->y);
        matrix_mul(&this->w, &this->x, &this->y_prime);
        relu_forward<<<this->output_dim, 1>>>(this->y.get_d(), this->output_dim);
        cudaDeviceSynchronize();

        return y;
    }

    Matrix<float> backward_first(Matrix<float> input_gradient, Matrix<float> x)
    {
        relu_backward<<<this->output_dim,1>>>(input_gradient.get_d(), this->y_prime.get_d(), this->dy.get_d(), this->output_dim);
        matrix_mul_ty(&this->dy, &x, &this->dw);
        cudaDeviceSynchronize();

        return dy;
    }
    Matrix<float> backward(Matrix<float> input_gradient, Matrix<float> x)
    {
        
    }

    void update()
    {
        subtract<<<w.get_len(), 1>>>(w.get_d(), dw.get_d(), dw.get_len());
        cudaDeviceSynchronize();
    }
};

// template<class float>
// class FullyConnectedLayer : public Layer
// {
//     private:
//         float /* data */
//     public:
//         FullyConnectedLayer(float /* args */);
// };

// template <class float>
// class Layers
// {
//     public:
//         Layers(/* args */);
//         ~Layers();
//         add(Layer<float> layer)
// };

// Layers::Layers(/* args */)
// {
// }

// Layers::~Layers()
// {
// }

class Network
{
  public:
    std::list<Layer *> layers;
    int input_dimension;
    int output_classes;
    int num_hidden;
    int *hidden_sizes;
    Network(int input_dimension, int output_classes, int num_hidden, int *hidden_sizes)
    {
        this->input_dimension = input_dimension;
        this->output_classes = output_classes;
        this->num_hidden = num_hidden;
        this->hidden_sizes = hidden_sizes;

        Layer *input_layer = new Layer(input_dimension, hidden_sizes[0]);
        this->layers.insert(layers.end(), input_layer);
        for (int i = 0; i < num_hidden - 1; i++)
        {
            Layer *hidden = new Layer(hidden_sizes[i], hidden_sizes[i + 1]);
            this->layers.insert(layers.end(), hidden);
        }
        Layer *output_layer = new Layer(hidden_sizes[num_hidden - 1], output_classes);
        this->layers.insert(layers.end(), output_layer);
    }

    Matrix<float> forward(Matrix<float> x)
    {   
        Matrix<float> temp;
        temp = Matrix<float>(this->input_dimension, 1, 1);
        temp = x;
        temp.to_gpu();
        int i = 0;
        for (std::list<Layer *>::iterator it = this->layers.begin(); it != this->layers.end(); ++it)
        {
            Layer *exec = *it;
            Matrix<float> temp2 = Matrix<float>(this->hidden_sizes[i],1,1);
            temp2 = (*exec).forward(temp);
            temp = Matrix<float>(this->hidden_sizes[i],1,1);
            temp = temp2;
            i++;
        }

        return temp;
    }

    void backpropagation(Matrix<float> loss)
    {   
        loss.to_gpu();
        (*(layers.end() - 1)).backward(loss, (*(layers.end() - 2))->y);
        for (std::list<Layer *>::iterator it = this->layers.end() - 2; it != this->layers.begin(); it--)
        {
            Layer *exec = *it;
            Matrix<float> temp2 = Matrix<float>(this->hidden_sizes[i],1,1);
            temp2 = (*exec).backward(temp);
            temp = Matrix<float>(this->hidden_sizes[i],1,1);
            temp = temp2;
            i++;
        }
    }
};