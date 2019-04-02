
#include <stdio.h>
#include <cublas_v2.h>

#include "matrix.h"

cublasHandle_t& get_cublass_handle()
{
    static cublasHandle_t cublass_handle_ = NULL;
	if(cublass_handle_ == NULL && (cublasCreate(&cublass_handle_) != CUBLAS_STATUS_SUCCESS)){
			printf ("get_cublass_handle(): cublasCreate failed");
			exit(0);
	}
	return cublass_handle_;
}

/**
 * z = x*y
 */
void matrix_mul(Matrix<float>* x, Matrix<float>* y, Matrix<float>*z)
{
    // error checking 
    if(x->n_cols != y->n_rows)
    {
        fprintf(stderr, "matrix_mul(): n_rows of y should be equal to the n_cols of x: %ld != %ld \n", x->n_rows, y->n_cols);
        exit(0);
    }
    if(z->n_cols != y->n_cols){
		fprintf(stderr, "matrix_mul(): n_cols of y and z don't match: %ld != %ld\n", y->n_cols, z->n_cols);
		exit(0);
    }
    if(x->n_rows != z->n_rows){
		fprintf(stderr, "matrix_mul(): n_rows of x and z don't match: %ld != %ld\n", x->n_rows, z->n_rows);
		exit(0);
    }
    if(x->n_channels != 1 || y->n_channels != 1 || z->n_channels != 1){
		fprintf(stderr, "matrix_mul(): 3D matrix multiplication is not allowed. One of the n_channels != 1\n");
		exit(0);
    }
    
	float a = 1.0, b = 0.0;
     
    cublasStatus_t cublas_status = cublasSgemm(get_cublass_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
		y->n_cols, x->n_rows, y->n_rows,
		&a, y->get_d(), y->n_cols,
		x->get_d(), x->n_cols, &b,
		z->get_d(), z->n_cols
    );
    
    checkCudaErrors(cudaStreamSynchronize(0))
	if(cublas_status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "matrix_mul(): cublasSgemm()\n");
        delete x;
        delete y;
		delete z;
		exit(0);
	}
}

/**
* z = T(x) * y
*/
void matrix_mul_tx(Matrix<float>* x, Matrix<float>*y, Matrix<float>*z)
{


	if(x->n_cols != z->n_rows || x->n_rows != y->n_rows || y->n_cols !=  z->n_cols){
		fprintf(stderr, "matrix_mul(): n_rows of x and z don't match: %ld != %ld\n", x->n_rows, z->n_rows);
		exit(0);
    }

    if(y->n_channels != 1 || x->n_channels != 1 ||  z->n_channels != 1){
		fprintf(stderr, "matrix_mul(): 3D matrix multiplication is not allowed. One of the n_channels != 1\n");
		exit(0);
	}

	float a = 1.0, b = 0.0;
    cublasStatus_t cublas_status = cublasSgemm(get_cublass_handle(), CUBLAS_OP_N, CUBLAS_OP_T, 
        y->n_cols, x->n_cols, y->n_rows,
		&a, y->get_d(), y->n_cols,
		x->get_d(), x->n_cols, &b,
		z->get_d(), z->n_cols);
        
    checkCudaErrors(cudaStreamSynchronize(0))
    if(cublas_status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "matrix_mul(): cublasSgemm()\n");
            delete x;
            delete y;
            delete z;
            exit(0);
    }
}

/**
* z = x * T(y)
*/
void matrix_mul_ty(Matrix<float>* x, Matrix<float>*y, Matrix<float>*z)
{

	if( x->n_rows !=  z->n_rows || x->n_cols != y->n_cols ||  y->n_rows !=  z->n_cols){
        fprintf(stderr, "matrix_mul(): n_rows of x and z don't match: %ld != %ld\n", x->n_rows, z->n_rows);
		exit(0);
    }

    if( y->n_channels != 1 || x->n_channels != 1 || z->n_channels != 1){
		fprintf(stderr, "matrix_mul(): 3D matrix multiplication is not allowed. One of the n_channels != 1\n");
		exit(0);
	}
    
	float a = 1.0, b = 0.0;
	cublasStatus_t cublas_status = cublasSgemm(get_cublass_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
		y->n_rows, x->n_rows, y->n_cols,
		&a, y->get_d(), y->n_cols,
		x->get_d(), x->n_cols, &b,
		z->get_d(), z->n_cols);

        checkCudaErrors(cudaStreamSynchronize(0))
        if(cublas_status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "matrix_mul(): cublasSgemm()\n");
            delete x;
            delete y;
            delete z;
            exit(0);
        }
}



// int main(){
    
//     Matrix<float> a = Matrix<float>(100,100,1);
//     a.to_gpu();

//     Matrix<float> b = Matrix<float>(100,100,1);
//     b.to_gpu();

//     Matrix<float> c = Matrix<float>(100,100,1);
//     c.to_gpu();

//     matrix_mul_tx(&a,&b,&c);

//     c.to_cpu();

//     printf(" %f \n", c.get(1,2,3));
    
//     int kk;
//     scanf("%d",&kk);
//     return 0;
// }