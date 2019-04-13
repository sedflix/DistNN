#include "../utils/matrix.h"
#include <cuda_runtime.h>

__global__ void relu_forward(float *x, int N);

__global__ void backward_w(float *input_grad, float *cache, float *output_grad, int N);
__global__ void backward_x(float *input_grad, float *cache, float *output_grad, int N);
__global__ void relu_backward(float *input_grad, float *cache, float *output_grad, int N);
__global__ void subtract(float *a, float *b, int N);
__global__ void add(float *a, float *b, int N);

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
        Matrix<float> dy_;

        Matrix<float> x;

        Layer(int input_dim, int output_dim) 
        {
            this->input_dim = input_dim;
            this->output_dim = output_dim;

            w = Matrix<float>(output_dim, input_dim, 1);
            dw = Matrix<float>(output_dim, input_dim, 1);
            
            x = Matrix<float>(input_dim, 1, 1);
            dx = Matrix<float>(input_dim, 1, 1);

            y = Matrix<float>(output_dim, 1, 1);
            dy_ = Matrix<float>(output_dim, 1, 1);

            b = Matrix<float>(output_dim, 1, 1);
            db = Matrix<float>(output_dim, 1, 1);
        }

        ~Layer() 
        {
            
        }

        Matrix<float> forward(Matrix<float> x) 
        {
            this->x = x;

            // this->x.to_gpu();
            // this->w.to_gpu();
            // this->y.to_gpu();

            matrix_mul(&this->w, &this->x, &this->y);
            relu_forward<<<this->output_dim,1>>>(this->y.get_d(),this->output_dim);
            cudaDeviceSynchronize();


            return y;
        }

       void backward(Matrix<float> dy) {
            // relu_backward<<<this->output_dim,1>>>(dy.get_d(), y.get_d(), this->dy_.get_d(), this->output_dim);
            matrix_mul_tx(&w, &dy, &dx);
            matrix_mul_ty(&dy, &x, &dw);
            cudaDeviceSynchronize();
        }

        void update() 
        {
            subtract<<<w.get_len(),1>>>(w.get_d(), dw.get_d(), dw.get_len());
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

