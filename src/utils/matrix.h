#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>


/*cublass helpers*/
cublasHandle_t& get_cublass_handle();

// citation: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define checkCudaErrors(ans) gpuAssert((ans), __FILE__, __LINE__);
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

template <class T>
class Matrix
{
  private:
    T *data_h; // host data
    T *data_d; // device data

  public:
    long n_rows, n_cols, n_channels;

  public:
    // constructor with the given data
    Matrix(long n_rows, long n_cols, long n_channels, T *data)
    {
        this->n_rows = n_rows;
        this->n_cols = n_cols;
        this->n_channels = n_channels;

        this->data_h = data;
        this->data_d = NULL;

        this->malloc_d();
    }

    Matrix() {
    
    }

    // default constructor
    Matrix(long n_rows, long n_cols, long n_channels)
    {
        this->n_rows = n_rows;
        this->n_cols = n_cols;
        this->n_channels = n_channels;

        data_h = NULL;
        data_d = NULL;

        this->malloc_h();
        this->malloc_d();
        this->to_gpu();
    }

    // let's destroy my life plox
    // ~Matrix()
    // {
    //     // free gpu memory
    //     if (this->data_d != NULL)
    //     {
    //         checkCudaErrors(cudaFree(this->data_d));
    //     }

    //     // free cpu memory
    //     if (this->data_h != NULL)
    //     {
    //         checkCudaErrors(cudaFreeHost(this->data_h));
    //     }
    // }

    // get the index in the corresponding 1-D array
    long get_index(long i, long j, long k)
    {
        return (i * this->n_cols + j) + this->n_cols * this->n_rows * k;
    }

    // gives the size of the correspoding 1-D array
    long get_len()
    {
        return this->n_cols * this->n_rows * this->n_channels;
    }

    // gives the actual size, in bytes, of the correspoding 1-D array
    long get_size()
    {
        return this->get_len() * sizeof(T);
    }

    // get the value at (i,j,k)
    T get(long i, long j, long k)
    {
        // this->malloc_h();
        return this->data_h[this->get_index(i, j, k)];
    }

    // set he value at (i,j,k)
    void set(long i, long j, long k, T value)
    {
        this->malloc_h();
        this->data_h[this->get_index(i, j, k)] = value;
    }

    // indexing operator
    T &operator()(long i, long j, long k)
    {   
        return this->get(i, j, k);
    }

    // set everything to 0 on cpu
    void reset_h()
    {
        // TODO: CHECK THIS ERROR
        for (int i = 0; i<this->get_size(); i++)
        {
            this->data_h[i] = float(std::rand() % 100) / 100;
            this->data_h[i] = 1;

        }
        // memset(this->data_h, 2.0, this->get_size());
    }

    // set everything to 0 on gpu
    void reset_d()
    {   
        // malloc avoided due to recursion problem
        checkCudaErrors(cudaMemset(this->data_d, 2, this->get_size()));
    }

    // get pointer to the cpu data
    T *&get_h()
    {
        this->malloc_h();
        return this->data_h;
    }

    // get pointer to the gpu data
    T *&get_d()
    {
        this->malloc_d();
        return this->data_d;
    }

    // let's copy stuffs from gpu to cpu
    void to_cpu()
    {
        this->malloc_d();
        this->malloc_h();
        checkCudaErrors(cudaMemcpy(this->data_h,
                                   this->data_d,
                                   this->get_size(),
                                   cudaMemcpyDeviceToHost));
    }

    // let's copy stuffs from cpu to gpu
    void to_gpu()
    {
        this->malloc_d();
        this->malloc_h();
        checkCudaErrors(cudaMemcpy(this->data_d, 
                                    this->data_h, 
                                    this->get_size(), 
                                    cudaMemcpyHostToDevice));
    }

    // let's copy stuffs from cpu to gpu using stream
    void to_gpu(cudaStream_t stream)
    {
        checkCudaErrors(cudaMemcpyAsync(this->data_d,
                                        this->data_h,
                                        this->get_size(),
                                        cudaMemcpyHostToDevice,
                                        stream));
    }

    // malloc data on host using parameters of the class
    void malloc_h()
    {
        if (data_h == NULL)
        {
            checkCudaErrors(cudaHostAlloc(&this->data_h, this->get_size(), cudaHostAllocPortable));
            if (!this->data_h)
            {
                fprintf(stderr, "Matrix::malloc_h() Unable to allocate %ld bytes on CPU \n", this->get_size());
            }
            this->reset_h();
        }
    }

    // malloc data on the GPU using parameters of the class
    void malloc_d()
    {
        if (data_d == NULL)
        {
            if (cudaMalloc((void **)&this->data_d, this->get_size()) != cudaSuccess)
            {
                fprintf(stderr, "Matrix::malloc_d() Unable to allocate %ld bytes on GPU \n", this->get_size());
                exit(0);
            }
        }
    }


    void print() 
    {
        for(int k = 0; k < this->n_channels; k++)
        {    
            for(int i = 0; i < this->n_rows; i++)
            {
                for(int j = 0; j < this->n_cols; j++)
                {
                    printf("%f, ", this->get(i,j,k));
                }
                printf("\n");
            }
            printf("\n");
        }
        
    }
};


/**
 * z = x*y
 */

void matrix_mul(Matrix<float>* x, Matrix<float>* y, Matrix<float>* z);
void matrix_mul_tx(Matrix<float>* x, Matrix<float>*y, Matrix<float>*z);
void matrix_mul_ty(Matrix<float>* x, Matrix<float>*y, Matrix<float>*z);