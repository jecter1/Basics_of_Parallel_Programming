#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
	double* data;
	int w, h;
} Matrix;

Matrix matrixCreate(int w, int h);

void matrixDelete(Matrix a);

void matrixCopy(Matrix from, Matrix to);

void matrixFillZeros(Matrix a);

void matrixMulScal(Matrix a, double k, Matrix res);

void matrixAdd(double k1, Matrix a, double k2, Matrix b, Matrix res);

void matrixMul(Matrix a, Matrix b, Matrix res);

void vectorMul(Matrix a, Matrix b, double* res);

#endif