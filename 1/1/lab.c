#include "../matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VECTOR_WIDTH 1

int CGM(Matrix A, Matrix b, Matrix x, const double eps, int* cnt);

int readInput(Matrix* A, Matrix* b, char* pathname);

int printOutput(Matrix x, char* pathname);

int main() {
	const double eps = 1e-5;
	Matrix A;
	Matrix b;

	readInput(&A, &b, "../input.txt");

	int cnt;
	Matrix x = matrixCreate(VECTOR_WIDTH, b.h);

	
	struct timespec tm_start, tm_finish;
	double best_res = 0;
	double res;
	
	for (int i = 0; i < 5; ++i) {
		clock_gettime(CLOCK_REALTIME, &tm_start);
		CGM(A, b, x, eps, &cnt);
		clock_gettime(CLOCK_REALTIME, &tm_finish);
	
		res = tm_finish.tv_sec - tm_start.tv_sec + 1e-9 * (tm_finish.tv_nsec - tm_start.tv_nsec);

		if (best_res == 0 || res < best_res) {
			best_res = res;
		}
	}

	printf("Iterations count: %d\n", cnt);
	printf("Time elapsed: %lf sec.\n", best_res);

	printOutput(x, "output.txt");

	matrixDelete(A);
	matrixDelete(b);
	matrixDelete(x);
	return 0;
}

int CGM(Matrix A, Matrix b, Matrix x, const double eps, int* cnt) {
	Matrix r = matrixCreate(VECTOR_WIDTH, b.h);
	Matrix z = matrixCreate(VECTOR_WIDTH, b.h);

	double alpha;
	double beta;

	// Temporary variables
	Matrix tmp_vector = matrixCreate(VECTOR_WIDTH, b.h);
	double tmp_scalar;

	// length_b = (b, b)
	double length_b;
	vectorMul(b, b, &length_b);

	// x[0] = (0 ... 0)
	matrixFillZeros(x);

	// r[0] = b[0] - A * x[0]
	matrixMul(A, x, tmp_vector);
	matrixAdd(1, b, -1, tmp_vector, r);

	// z[0] = r[0]
	matrixCopy(r, z);

	// length_r = (r[n], r[n])
	double length_r;
	vectorMul(r, r, &length_r);
	// length_r_next = (r[n + 1], r[n + 1])
	double length_r_next;

	*cnt = 0;
	do {
		// alpha[n + 1] = (r[n], r[n]) / (A * z[n], z[n])
		matrixMul(A, z, tmp_vector);
		vectorMul(tmp_vector, z, &tmp_scalar);
		alpha = length_r / tmp_scalar;

		// x[n + 1] = x[n] + alpha[n + 1] * z[n]
		matrixAdd(1, x, alpha, z, x);

		// r[n + 1] = r[n] - alpha[n + 1] * A * z[n]
		matrixAdd(1, r, -alpha, tmp_vector, r);
		vectorMul(r, r, &length_r_next);

		// beta[n + 1] = (r[n + 1], r[n + 1]) / (r[n], r[n])
		beta = length_r_next / length_r;
		length_r = length_r_next;

		// z[n + 1] = r[n + 1] + beta[n + 1] * z[n]
		matrixAdd(1, r, beta, z, z);

		++(*cnt);
	} while (length_r > length_b * eps * eps);

	matrixDelete(tmp_vector);
	matrixDelete(r);
	matrixDelete(z);

	return 0;
}

int readInput(Matrix* A, Matrix* b, char* pathname) {
	FILE* fin = fopen(pathname, "r");
	int N;
	fscanf(fin, "%d", &N);

	*A = matrixCreate(N, N);
	*b = matrixCreate(VECTOR_WIDTH, N);

	for(int i = 0; i < N * N; ++i) {
		fscanf(fin, "%lf", A->data + i);
	}

	for(int i = 0; i < N; ++i) {
		fscanf(fin, "%lf", b->data + i);
	}

	fclose(fin);
	return 0;
}

int printOutput(Matrix x, char* pathname) {
	FILE* fout = fopen(pathname, "w+");

	for (int i = 0; i < x.w * x.h; ++i) {
		fprintf(fout, "%lf ", x.data[i]);
	}
	fprintf(fout, "\n");

	fclose(fout);
	return 0;
}