#ifndef PARALLEL_H
#define PARALLEL_H

#include "../matrix.h"

void matrixVectorParallelMul(int size, int rank, Matrix A, Matrix b, Matrix res); // (b.w == 1)!!!

void vectorParallelMul(int size, int rank, Matrix a, Matrix b, double* res);

#endif