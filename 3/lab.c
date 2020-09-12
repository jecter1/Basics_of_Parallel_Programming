#include <stdio.h>
#include "functions.h"

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);

	int p1, p2, n1, n2, n3, m1, m2, m3;
	Read_input(argc, argv, &p1, &p2, &n1, &n2, &n3, &m1, &m2, &m3);

	MPI_Comm comm_row, comm_col;
	Create_grid_communicators(MPI_COMM_WORLD, p1, p2, &comm_row, &comm_col);

	int rank_col, rank_row, rank_col_main, rank_row_main;
	Calculate_grid_ranks(comm_row, comm_col, &rank_row, &rank_col, &rank_row_main, &rank_col_main);

	const int iterations = 5;
	double best_time = 0;
	double* C;
	for (int i = 0; i < iterations; ++i) {
		C = NULL;

		double start_time = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
		Use_algorithm(n1, n2, n3, m1, m2, m3, rank_col_main, rank_row_main, rank_col, rank_row, comm_col, comm_row, &C);
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
		printf("Best time of %d iteration(s): %lf sec.\n", iterations, best_time);

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