#include "layers/layers.h"

int main() {
    Layer a = Layer(100,10);
    Matrix<float> x = Matrix<float>(1,100,1);
    a.forward(x);
}
