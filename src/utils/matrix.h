#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"




    template <class T>
    class Matrix
    {
        private:
            T *data_h; // host data
            T *data_d; // device data

        public:
            long rows, cols, channels; 
        public:

            // constructor with the given data
            Matrix(long rows, long cols, long channels, T* data)
            {
                Matrix(rows, cols, channels);
                malloc_h();
            }

            // default constructor
            Matrix(long rows, long cols, long channels) 
            {
                    this->rows = rows;
                    this->cols = cols;
                    this->channels = channels;

                    data_h = NULL;
                    data_d = NULL;
            }

            // let's destroy my life plox
            ~Matrix() 
            {   
                // free gpu memory
                if (this->data_d != NULL)
                {
                    cudaFree(this->data_d);
                }
                
                // free cpu memory
                if (this->data_h != NULL)
                {
                    cudaFreeHost(this->data_h);
                }
            }

            // get the index in the corresponding 1-D array
            long get_index(long i, long j, long k) 
            {
                return (i * this->cols + j) + this->cols * this->rows * k;
            }


            // gives the size of the correspoding 1-D array
            long get_len()
            {
                return this->cols * this->rows * this->channels;
            } 

            // gives the actual size, in bytes, of the correspoding 1-D array
            long get_size()
            {
                return this->get_len() * sizeof(*data_h);
            }

            // get the value at (i,j,k)
            T get(long i, long j, long k) 
            {
                this->malloc_h();
                return this->data_h[this->get_index(i,j,k)];
            }

            // set he value at (i,j,k)
            void set(long i, long j, long k, T value) 
            {
                this->malloc_h();
                this->data_h[this->get_index(i,j,k)] = value;
            }
            
            // indexing operator
            T& operator()(long i, long j, long k) 
            {
                return this->get(i,j,k);
            }

            // set everything to 0 on cpu
            void reset_h()
            {   
                memset(data_h, 0, this->get_size());  
            }

            // set everything to 0 on gpu
            void reset_d()
            {
                if(cudaMemset(data_h, 0, this->get_size()) != cudaSuccess) 
                {
                    fprintf(stderr, "Matrix::malloc_d() Unable to memset %ld bytes on GPU \n", this->get_size());
                    exit(0);
                } 
            }

            // get pointer to the cpu data
            T*& get_h()
            {
                this->malloc_h();
                return this->data_h;
            }
            
            // get pointer to the gpu data
            T*& get_d()
            {
                this->malloc_d();
                return this->data_d;
            }

            // let's copy stuffs from gpu to cpu
            void to_cpu(){
                this->malloc_d();
                this->malloc_h();
                checkCudaErrors(cudaMemcpy(this->data_h, 
                this->data_d, 
                this->get_size(), 
                cudaMemcpyDeviceToHost));
            }

            // let's copy stuffs from cpu to gpu
            void to_gpu(){
                this->malloc_d();
                this->malloc_h();
                checkCudaErrors((cudaMemcpy(this->data_d, 
                this->data_h, 
                this->get_size(), 
                cudaMemcpyHostToDevice));
            }

            // let's copy stuffs from cpu to gpu using stream
            void to_gpu(cudaStream_t stream){
                checkCudaErrors(cudaMemcpyAsync(this->data_d, 
                this->data_h, 
                this->get_size(), 
                cudaMemcpyHostToDevice, 
                stream));
            }
            
            // malloc data on host using parameters of the class
            void malloc_h() 
            {
                if(data_h == NULL) {
                    cudaHostAlloc(this->data_h, this->get_size, cudaHostAllocPortable);
                    if(!this->data_h){
                        fprintf(stderr, "Matrix::malloc_h() Unable to allocate %ld bytes on CPU \n", this->get_size());
                    }
                    this->reset_h();               
                }
            }

            // malloc data on the GPU using parameters of the class
            void malloc_d() 
            {
                if(data_d == NULL) {
                    if(cudaMalloc(this->data_d, this->get_size()) != cudaSuccess)
                    {
                        fprintf(stderr, "Matrix::malloc_d() Unable to allocate %ld bytes on GPU \n", this->get_size());
                        exit(0);
                    }
                    this->reset_d();             
                }
            }
    };


