#include "matrix.h"
#include <stdio.h>

int main(){
    
    Matrix<float> b = Matrix<float>(1000,1000,1000);
    b.reset_h();
    b.set(1,2,3, 1.2);
    b.to_gpu();
    b.to_cpu();
    printf(" %f \n", b.get(1,2,3));
    
    int kk;
    scanf("%d",&kk);
    return 0;
}