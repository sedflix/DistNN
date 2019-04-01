#include <stdio.h>
#include <cublas_v2.h>

#include "matrix.h"

cublasHandle_t& get_cublass_handle()
{
	if(cublass_handle == NULL && (cublasCreate(&cublass_handle) != CUBLAS_STATUS_SUCCESS)){
			printf ("get_cublass_handle(): cublasCreate failed");
			exit(0);
	}
	return cublass_handle;
}

/**
 * z = x*y
 */
void matrix_mul(Matrix<float>* x, Matrix<float>* y, Matrix<float>*z)
{
	if(x->channels != 1 || y->channels != 1 || z->channels != 1){
		printf("matrix_mul(): 3D matrix multiplication is not allowed. One of the channels != 1\n");
		exit(0);
	}
	if(x->cols != y->rows || z->rows != x->rows || z->cols != y->cols){
		printf("matrix mul chanels != 1\n");
		exit(0);
    }
    
	float alpha = 1.0;
	float beta = 0.0;
     
    cublasStatus_t cublas_status = cublasSgemm(
		get_cublass_handle(), 
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		y->cols,
		x->rows,
		y->rows,
		&alpha,
		y->get_d(),
		y->cols,
		x->get_d(),
		x->cols,
		&beta,
		z->get_d(),
		z->cols);
    
    cudaStreamSynchronize(0);
	if(cublas_status != CUBLAS_STATUS_SUCCESS) {
        
        fprintf(stderr, "matrix_mul(): cublasSgemm()\n");
		cudaFree(x->get_d());
		cudaFree(y->get_d());
		cudaFree(z->get_d());
		exit(0);
	}
}


int main(){
    
    Matrix<float> a = Matrix<float>(100,100,1);
    a.to_gpu();

    Matrix<float> b = Matrix<float>(100,100,1);
    b.to_gpu();

    Matrix<float> c = Matrix<float>(100,100,1);
    c.to_gpu();

    matrix_mul(&a,&b,&c);

    c.to_cpu();

    printf(" %f \n", c.get(1,2,3));
    
    int kk;
    scanf("%d",&kk);
    return 0;
}