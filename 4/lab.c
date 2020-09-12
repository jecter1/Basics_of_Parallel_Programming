#include "functions.h"

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);

	const int MAIN_RANK = 0;
	const int ITERATIONS_CNT = 1;
	
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	double a = 10e+5;
	double eps = 1e-8;

	double k = 1000.0;

	double Dx = 2.0 * k;
	double Dy = 2.0 * k;
	double Dz = 2.0 * k;

	double x0 = -1.0 * k;
	double y0 = -1.0 * k;
	double z0 = -1.0 * k;

	int Nx, Ny, Nz;
	Read_arguments_values(argc, argv, &Nx, &Ny, &Nz);

	double* result;
	double max_delta;
	int algorithm_iterations_count;

	double iteration_time, start_time, best_time = 0.0;
	for (int i = 0; i < ITERATIONS_CNT; ++i) {
		result = NULL;

		if (rank == MAIN_RANK) {
			start_time = MPI_Wtime();
		}

		MPI_Barrier(MPI_COMM_WORLD);
		Use_Jacobi_method(rank, size, a, eps, Dx, Dy, Dz, x0, y0, z0, Nx, Ny, Nz, &result, &max_delta, &algorithm_iterations_count);
		MPI_Barrier(MPI_COMM_WORLD);

		if (rank == MAIN_RANK) {
			iteration_time = MPI_Wtime() - start_time;
			best_time = (iteration_time < best_time || best_time == 0.0) ? iteration_time : best_time;
		}

		if (i < ITERATIONS_CNT - 1) {
			free(result);
		}
	}

	free(result);
	if (rank == MAIN_RANK) {
		printf("Best time of %d iteration(s): %lf sec.\n", ITERATIONS_CNT, best_time);
		printf("Iterations in the algorithm: %d\n", algorithm_iterations_count);
		if (max_delta < 100 * eps) {
			printf("Good result\n");
		} else {
			printf("Bad result\n");
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}