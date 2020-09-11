#include <stdio.h>
#include "functions.h"

void execute(int n1, int n2, int n3, int m1, int m2, int m3,
	int rank_col_main, int rank_row_main, int rank_col, int rank_row,
	MPI_Comm comm_col, MPI_Comm comm_row, double** C);

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);

	int p1, p2, n1, n2, n3, m1, m2, m3;
	Read_input(argc, argv, &p1, &p2, &n1, &n2, &n3, &m1, &m2, &m3);

	MPI_Comm comm_row, comm_col;
	Create_grid_communicators(MPI_COMM_WORLD, p1, p2, &comm_row, &comm_col);

	int rank_col, rank_row, rank_col_main, rank_row_main;
	Calculate_grid_ranks(comm_row, comm_col, &rank_row, &rank_col, &rank_row_main, &rank_col_main);

	const int iterations = 10;
	double best_time = 0;
	double* C;
	for (int i = 0; i < iterations; ++i) {
		C = NULL;

		double start_time = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
		execute(n1, n2, n3, m1, m2, m3, rank_col_main, rank_row_main, rank_col, rank_row, comm_col, comm_row, &C);
		double finish_time = MPI_Wtime();

		if (i < iterations - 1) {
			free(C);
		}

		if ((finish_time - start_time < best_time) || best_time == 0) {
			best_time = finish_time - start_time;
		}
	}
	
	MPI_Comm_free(&comm_col);
	MPI_Comm_free(&comm_row);

	// Print result to "out.txt"
	if ((rank_row == rank_row_main) && (rank_col == rank_col_main)) {
		printf("Best time of %d iterations: %lf sec.\n", iterations, best_time);

		FILE* fout = fopen("out.txt", "w");

		for (int i = 0; i < n1; ++i) {
			for (int j = 0; j < n3; ++j) {
				fprintf(fout, "%lf\t", C[i * n3 + j]);
			}
			fprintf(fout, "\n");
		}

		free(C);
		fclose(fout);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}

void execute(int n1, int n2, int n3, int m1, int m2, int m3,
	int rank_col_main, int rank_row_main, int rank_col, int rank_row,
	MPI_Comm comm_col, MPI_Comm comm_row, double** C) {

	// Create matrices A and B on process with coordinates (0, 0)
	double *A = NULL, *B = NULL;
	if ((rank_col == rank_col_main) && (rank_row == rank_row_main)) {
		Create_filled_matrix(n1, n2, &A);
		Create_filled_matrix(n2, n3, &B);
	}

	// Scatter matrix A in column #0
	double* subA = (double*)malloc(sizeof(double) * m1 * m2);
	if (rank_row == rank_row_main) {
		Scatter_horizontal_stripes(A, n1, n2, subA, rank_col_main, comm_col);

		if (rank_col == rank_col_main) { 
			free(A);
		}
	}

	// Scatter matrix B in row #0
	double* subB = (double*)malloc(sizeof(double) * m2 * m3);
	if (rank_col == rank_col_main) {
		Scatter_vertical_stripes(B, n2, n3, subB, rank_row_main, comm_row);
		
		if (rank_row == rank_row_main) {
			free(B);
		}
	}

	// Broadcast matrix subA in rows from column #0
	MPI_Bcast(subA, m1 * m2, MPI_DOUBLE, rank_row_main, comm_row);

	// Broadcast matrix subB in columns from row #0
	MPI_Bcast(subB, m2 * m3, MPI_DOUBLE, rank_col_main, comm_col);

	// Calculate matrix subC on each process
	double* subC = (double*)malloc(sizeof(double) * m1 * m3);
	Multiplicate_matrices(m1, m2, m3, subA, subB, subC);

	free(subA);
	free(subB);

	// Gather result on process with coordinates (0, 0)
	if ((rank_row == rank_row_main) && (rank_col == rank_col_main)) {
		*C = (double*)malloc(sizeof(double) * n1 * n3);
	}

	Gather_result(subC, m1, m3, *C, rank_col_main, rank_row_main, comm_col, comm_row);

	free(subC);
}