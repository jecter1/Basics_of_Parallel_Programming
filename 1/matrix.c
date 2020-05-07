#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

Matrix matrixCreate(int w, int h) {
	Matrix res;
	
	res.data = (double*) malloc(sizeof(double) * w * h);
	res.w = w;
	res.h = h;
	
	return res;
}

void matrixDelete(Matrix a) {
	free(a.data);
}

void matrixCopy(Matrix from, Matrix to) {
	for(int i = 0; i < to.h * to.w; ++i) {
		to.data[i] = from.data[i];
	}
}

void matrixFillZeros(Matrix a) {
	for(int i = 0; i < a.h * a.w; ++i) {
		a.data[i] = 0;
	}
}

void matrixMulScal(Matrix a, double k, Matrix res) {
	for (int i = 0; i < a.h * a.w; ++i) {
		res.data[i] = a.data[i] * k;
	}
}

void matrixAdd(double k1, Matrix a, double k2, Matrix b, Matrix res) {
	for (int i = 0; i < a.h * a.w; ++i) {
		res.data[i] = k1 * a.data[i] + k2 * b.data[i];
	}
}

void matrixMul(Matrix a, Matrix b, Matrix res) {
	for (int i = 0; i < res.h; ++i) {
		for (int j = 0; j < res.w; ++j) {
			res.data[i * res.w + j] = 0;
			for (int k = 0; k < a.w; ++k) {
				res.data[i * res.w + j] += a.data[i * a.w + k] * b.data[k * b.w + j];
			}
		}
	}
}

void vectorMul(Matrix a, Matrix b, double* res) {
	*res = 0;
	for (int i = 0; i < a.h; ++i) {
		*res += a.data[i] * b.data[i];
	}
}