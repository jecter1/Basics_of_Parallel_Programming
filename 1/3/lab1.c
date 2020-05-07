#include "../matrix.h"
#include "parallel.h"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define VECTOR_WIDTH 1
#define MAIN_PROC 0

#define MIN_CNT 0
#define MAX_CNT 10

int CGM(int size, int rank, Matrix A, Matrix b, Matrix x, const double eps, int* cnt);

void readInput(int size, int rank, Matrix* A, Matrix* b, char* pathname);

int printOutput(int size, int rank, int N, Matrix x, char* pathname);

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	int world_size;
	int world_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	const double eps = 1e-5;
	Matrix A;
	Matrix b;

	readInput(world_size, world_rank, &A, &b, "../input.txt");

	int cnt;
	Matrix x = matrixCreate(VECTOR_WIDTH, b.h);

	double tm_start = MPI_Wtime();
	CGM(world_size, world_rank, A, b, x, eps, &cnt);
	double tm_finish = MPI_Wtime();

	if (world_rank == MAIN_PROC) {
		printf("Iterations count: %d\n", cnt);
		double time_elapsed = tm_finish - tm_start;
		printf("Time elapsed: %lf sec.\n", time_elapsed);
	}

	printOutput(world_size, world_rank, A.h, x, "output.txt");

	matrixDelete(A);
	matrixDelete(b);
	matrixDelete(x);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}

int CGM(int size, int rank, Matrix A, Matrix b, Matrix x, const double eps, int* cnt) {
	Matrix r = matrixCreate(VECTOR_WIDTH, b.h);
	Matrix z = matrixCreate(VECTOR_WIDTH, b.h);

	double alpha;
	double beta;

	// Temporary variables
	Matrix tmp_vector = matrixCreate(VECTOR_WIDTH, b.h);
	double tmp_scalar;

	// length_b = (b, b)
	double length_b;
	vectorParallelMul(size, rank, b, b, &length_b);

	// x[0] = (0 ... 0)
	matrixFillZeros(x);

	// r[0] = b[0] - A * x[0] <=> r[0] = b[0]
	matrixCopy(b, r);

	// z[0] = r[0]
	matrixCopy(r, z);

	// length_r = (r[n], r[n])
	double length_r;
	vectorParallelMul(size, rank, r, r, &length_r);
	// length_r_next = (r[n + 1], r[n + 1])
	double length_r_next;

	*cnt = 0;
	do {
		// alpha[n + 1] = (r[n], r[n]) / (A * z[n], z[n])
		matrixVectorParallelMul(size, rank, A, z, tmp_vector);
		vectorParallelMul(size, rank, tmp_vector, z, &tmp_scalar);
		alpha = length_r / tmp_scalar;

		// x[n + 1] = x[n] + alpha[n + 1] * z[n]
		matrixAdd(1, x, alpha, z, x);

		// r[n + 1] = r[n] - alpha[n + 1] * A * z[n]
		matrixAdd(1, r, -alpha, tmp_vector, r);
		vectorParallelMul(size, rank, r, r, &length_r_next);

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

void readInput(int size, int rank, Matrix* A, Matrix* b, char* pathname) {
	FILE* fin = NULL;

	int N;
	// READING & SENDING N
	if (rank == MAIN_PROC) {
		fin = fopen(pathname, "r");
		fscanf(fin, "%d", &N);
	}

	MPI_Bcast(
		/* buffer 	= */ &N, 
		/* count 	= */ 1, 
		/* datatype = */ MPI_INT, 
		/* root 	= */ MAIN_PROC, 
		/* comm 	= */ MPI_COMM_WORLD);
	// END


	// READING & SENDING PARTS OF A
	double* A_full = NULL;
	
	if (rank == MAIN_PROC) {
		A_full = (double*) malloc(sizeof(double) * N * N);
		for (int i = 0; i < N * N; ++i) {
			fscanf(fin, "%lf", A_full + i);
		}
	}

	int sendcounts[size];
	for (int i = 0; i < size; ++i) {
		sendcounts[i] = N / size;
		if (i < N % size) {
			sendcounts[i]++;
		}
		sendcounts[i] *= N;
	}

	int displs[size];
	displs[0] = 0;
	for (int i = 1; i < size; ++i) {
		displs[i] = displs[i - 1] + sendcounts[i - 1];
	}

	double* tmp = (double*) malloc(sizeof(double) * sendcounts[rank]);

	MPI_Scatterv(
		/* sendbuf 		= */ A_full, 
		/* sendcounts 	= */ sendcounts, 
		/* displs 		= */ displs, 
		/* sendtype 	= */ MPI_DOUBLE, 
		/* recvbuf 		= */ tmp, 
		/* recvcount 	= */ sendcounts[rank], 
		/* recvtype 	= */ MPI_DOUBLE, 
		/* root 		= */ MAIN_PROC, 
		/* comm 		= */ MPI_COMM_WORLD);

	if (rank == MAIN_PROC) {
		free(A_full);
	}

	A->w = sendcounts[rank] / N;
	A->h = N;
	A->data = (double*) malloc(sizeof(double) * sendcounts[rank]);

	for (int i = 0; i < A->w; ++i) {
		for (int j = 0; j < A->h; ++j) {
			A->data[j * A->w + i] = tmp[i * A->h + j];
		}
	}

	free(tmp);
	// END


	// READING & SENDING b
	double* b_full = NULL;	

	if (rank == MAIN_PROC) {
		b_full = (double*) malloc(sizeof(double) * N);
		for (int i = 0; i < N ; ++i) {
			fscanf(fin, "%lf", b_full + i);
		}
	}

	for (int i = 0; i < size; ++i) {
		sendcounts[i] /= N;
	}
	for (int i = 1; i < size; ++i) {
		displs[i] = displs[i - 1] + sendcounts[i - 1];
	}

	b->data = (double*) malloc(sizeof(double) * sendcounts[rank]);

	MPI_Scatterv(
		/* sendbuf 		= */ b_full, 
		/* sendcounts 	= */ sendcounts, 
		/* displs 		= */ displs, 
		/* sendtype 	= */ MPI_DOUBLE, 
		/* recvbuf 		= */ b->data, 
		/* recvcount 	= */ sendcounts[rank], 
		/* recvtype 	= */ MPI_DOUBLE, 
		/* root 		= */ MAIN_PROC, 
		/* comm 		= */ MPI_COMM_WORLD);

	if (rank == MAIN_PROC) {
		free(b_full);
	}

	b->w = VECTOR_WIDTH;
	b->h = sendcounts[rank];
	// END

	if (rank == MAIN_PROC) {
		fclose(fin);
	}
}

int printOutput(int size, int rank, int N, Matrix x, char* pathname) {
	double* result = NULL;

	if (rank == MAIN_PROC) {
		result = (double*) malloc(sizeof(double) * N);
	}

	int recvcounts[size];
	for (int i = 0; i < size; ++i) {
		recvcounts[i] = N / size;
		if (i < N % size) {
			recvcounts[i]++;
		}
	}
	int displs[size];
	displs[0] = 0;
	for (int i = 1; i < size; ++i) {
		displs[i] = displs[i - 1] + recvcounts[i - 1];
	}

	MPI_Gatherv(
		/* sendbuf 		= */ x.data,
		/* sendcount 	= */ x.h,
		/* sendtype 	= */ MPI_DOUBLE,
		/* recvbuf 		= */ result,
		/* recvcounts 	= */ recvcounts,
		/* displs 		= */ displs,
		/* recvtype 	= */ MPI_DOUBLE,
		/* root 		= */ MAIN_PROC,
		/* comm 		= */ MPI_COMM_WORLD);

	if (rank == MAIN_PROC) {
		FILE* fout = fopen(pathname, "w+");

		for (int i = 0; i < N; ++i) {
			fprintf(fout, "%lf ", result[i]);
		}
		fprintf(fout, "\n");

		free(result);
		fclose(fout);
	}


	return 0;
}