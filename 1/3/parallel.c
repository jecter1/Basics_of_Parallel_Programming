#include "parallel.h"
#include <mpi.h>
#include <stdlib.h>

#define MAIN_PROC 0
#define VECTOR_WIDTH 1

void matrixVectorParallelMul(int size, int rank, Matrix A, Matrix b, Matrix res) {
	Matrix sub_res = matrixCreate(VECTOR_WIDTH, A.h);
	matrixMul(A, b, sub_res);

	double* full_res = NULL;
	if (rank == MAIN_PROC) {
		full_res = (double*) malloc(sizeof(double) * A.h);
	}

	MPI_Reduce(
		/* sendbuf 		= */ sub_res.data,
		/* recvbuf 		= */ full_res,
		/* count 		= */ A.h,
		/* datatype 	= */ MPI_DOUBLE,
		/* op 			= */ MPI_SUM,
		/* root			= */ MAIN_PROC,
		/* comm 		= */ MPI_COMM_WORLD);

	int sendcounts[size];
	for (int i = 0; i < size; ++i) {
		sendcounts[i] = A.h / size;
		if (i < A.h % size) {
			sendcounts[i]++;
		}
	}
	int displs[size];
	displs[0] = 0;
	for (int i = 1; i < size; ++i) {
		displs[i] = displs[i - 1] + sendcounts[i - 1];
	}

	MPI_Scatterv(
		/* sendbuf 		= */ full_res, 
		/* sendcounts 	= */ sendcounts, 
		/* displs 		= */ displs, 
		/* sendtype 	= */ MPI_DOUBLE, 
		/* recvbuf 		= */ res.data, 
		/* recvcount 	= */ sendcounts[rank], 
		/* recvtype 	= */ MPI_DOUBLE, 
		/* root 		= */ MAIN_PROC, 
		/* comm 		= */ MPI_COMM_WORLD);

	matrixDelete(sub_res);
	if (rank == 0) {
		free(full_res);
	}
}

void vectorParallelMul(int size, int rank, Matrix a, Matrix b, double* res) {
	double sub_res;
	vectorMul(a, b, &sub_res);

	MPI_Allreduce(
		/* sendbuf 		= */ &sub_res,
		/* recvbuf 		= */ res,
		/* count 		= */ 1,
		/* datatype 	= */ MPI_DOUBLE,
		/* op 			= */ MPI_SUM,
		/* comm 		= */ MPI_COMM_WORLD);
}