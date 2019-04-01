#include <stdio.h>
#include <stdlib.h>

namespace cu
{ 

    template <class T>
    class Matrix
    {   

        public:

            long rows, cols, channels;

            T *data_h; // host data
            T *data_d; // device data

            // constructor with the given data
            Matrix(long rows, long cols, long channels, T* data);

            // default constructor
            Matrix(long rows, long cols, long channels);

            // let's destroy my life plox
            ~Matrix();

            // get the index in the corresponding 1-D array
            long get_index(long i, long j, long k);

            // gives the size of the correspoding 1-D array
            long get_len(); 

            // gives the actual size, in bytes, of the correspoding 1-D array
            long get_size();

            // get the value at (i,j,k)
            T get(long i, long j, long k);

            // set he value at (i,j,k)
            void set(long i, long j, long k, T value);
            
            // indexing operator
            T& operator()(long i, long j, long k);

            // set everything to 0 on cpu
            void reset_h();

            // set everything to 0 on gpu
            void reset_d();

            // get pointer to the cpu data
            T*& get_h();
            
            // get pointer to the gpu data
            T*& get_d();

            // let's copy stuffs from gpu to cpu
            void to_cpu();

            // let's copy stuffs from cpu to gpu
            void to_gpu();

            // let's copy stuffs from cpu to gpu using stream
            void to_gpu(int stream);

            // malloc data on host using parameters of the class
            void malloc_h();

            // malloc data on the GPU using parameters of the class
            void malloc_d();
    };
}

