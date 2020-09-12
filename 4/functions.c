#include "functions.h"
#include <unistd.h>

void Read_arguments_values(int argc, char* argv[], int* Nx, int* Ny, int* Nz) {
	const int DIMS = 4;
	const int MAIN_RANK = 0;
	const int MIN_VALUE = 1;

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (argc != DIMS) {
		if (rank == MAIN_RANK) {
			fprintf(stderr, "Usage: mpirun -np 'proc_num' %s 'Nx' 'Ny' 'Nz'\n", argv[0]);
		}
		MPI_Abort(MPI_COMM_WORLD, -1); // incorrect usage
	}

	*Nx = atoi(argv[1]);
	*Ny = atoi(argv[2]);
	*Nz = atoi(argv[3]);

	if ((*Nx) < MIN_VALUE || (*Ny) < MIN_VALUE || (*Nz) < MIN_VALUE) {
		if (rank == MAIN_RANK) {
			fprintf(stderr, "Usage: mpirun -np 'proc_num' %s 'Nx' 'Ny' 'Nz'\n", argv[0]);
		}
		MPI_Abort(MPI_COMM_WORLD, -2); // incorrect usage
	}
}

double Calculate_phi(double x, double y, double z) {
	return (x * x + y * y + z * z);
}

double Calculate_ro(double a, double x, double y, double z) {
	return (6 - a * Calculate_phi(x, y, z));
}

void Calculate_layers(int rank, int size, int Nx, int* count, int* first, int* last) {
	*count = Nx / size;
	*first = rank * (*count);

	if (Nx % size != 0) {
		if (rank < Nx % size) {
			*count += 1;
			*first += rank;
		} else {
			*first += Nx % size;
		}
	}

	*last = (*first) + (*count) - 1;
}

double Approximate_phi_value(double a, double Hx, double Hy, double Hz, 
	double phi_nextX, double phi_nextY, double phi_nextZ, 
	double phi_prevX, double phi_prevY, double phi_prevZ,
	double x, double y, double z) {

	double C = 1.0 / (2.0 / (Hx * Hx) + 2.0 / (Hy * Hy) + 2.0 / (Hz * Hz) + a);

	double phiX = (phi_nextX + phi_prevX) / (Hx * Hx);
	double phiY = (phi_nextY + phi_prevY) / (Hy * Hy);
	double phiZ = (phi_nextZ + phi_prevZ) / (Hz * Hz);

	double Ro = Calculate_ro(a, x, y, z);

	return C * (phiX + phiY + phiZ - Ro);
}

void Approximate_layer(int layer_num, int layer_first, int layer_last, int Ny, int Nz, 
	double* phi_values, const double* prev_layer, const double* next_layer,
	double a, double Hx, double Hy, double Hz,
	double x0, double y0, double z0, double* max_delta) {

	double phi_nextX, phi_nextY, phi_nextZ;
	double phi_prevX, phi_prevY, phi_prevZ;

	double x, y, z;

	double delta;

	int layer_num_process = layer_num - layer_first;
	int layers_count_process = layer_last - layer_first + 1;

	x = x0 + layer_num * Hx;
	for (int j = 1; j < Ny - 1; ++j) {
		y = y0 + j * Hy;
		for (int k = 1; k < Nz - 1; ++k) {
			z = z0 + k * Hz;

			if (layer_num == layer_first && layer_num == layer_last) {
				phi_nextX = next_layer[j * Nz + k];
				phi_prevX = prev_layer[j * Nz + k];
			} else if (layer_num == layer_first && layer_num != layer_last) {
				phi_nextX = phi_values[(layer_num_process + 1) * Ny * Nz + j * Nz + k];
				phi_prevX = prev_layer[j * Nz + k];
			} else if (layer_num == layer_last && layer_num != layer_first) {
				phi_nextX = next_layer[j * Nz + k];
				phi_prevX = phi_values[(layer_num_process - 1) * Ny * Nz + j * Nz + k];
			} else {
				phi_nextX = phi_values[(layer_num_process + 1) * Ny * Nz + j * Nz + k];
				phi_prevX = phi_values[(layer_num_process - 1) * Ny * Nz + j * Nz + k];
			}

			phi_nextY = phi_values[layer_num_process * Ny * Nz + (j + 1) * Nz + k];
			phi_prevY = phi_values[layer_num_process * Ny * Nz + (j - 1) * Nz + k];

			phi_nextZ = phi_values[layer_num_process * Ny * Nz + j * Nz + (k + 1)];
			phi_prevZ = phi_values[layer_num_process * Ny * Nz + j * Nz + (k - 1)];

			delta = phi_values[layer_num_process * Ny * Nz + j * Nz + k];

			phi_values[layer_num_process * Ny * Nz + j * Nz + k] = 
				Approximate_phi_value(
					a, Hx, Hy, Hz, 
					phi_nextX, phi_nextY, phi_nextZ, 
					phi_prevX, phi_prevY, phi_prevZ, 
					x, y, z);

			delta -= phi_values[layer_num_process * Ny * Nz + j * Nz + k];

			delta = (delta < 0) ? -delta : delta;

			if (*max_delta < delta) {
				*max_delta = delta;
			}
		}
	}
}

void Use_Jacobi_method(int rank, int size, 
	double a, double eps, 
	double Dx, double Dy, double Dz, 
	double x0, double y0, double z0,
	int Nx, int Ny, int Nz,
	double** phi_values, double* max_delta,
	int* iterations) {

	// Hx = r(x[i-1], x[i]), Hy = r(y[i-1], y[i]), Hz = r(z[i-1], z[i]) 
	double Hx = Dx / (Nx - 1);
	double Hy = Dy / (Ny - 1);
	double Hz = Dz / (Nz - 1);

	// Calculate length (size_x) of grid for each process
	int layers_count;
	int layer_first;
	int layer_last;
	Calculate_layers(rank, size, Nx, &layers_count, &layer_first, &layer_last);

	// Create arrays for recording approximate value of the function in each node
	const int LAYER_SIZE = Ny * Nz;

	*phi_values = (double*)malloc(sizeof(double) * layers_count * LAYER_SIZE);

	const int FIRST_RANK = 0;
	const int LAST_RANK = size - 1;

	double* prev_layer = NULL; // array to receive values from previous process
	if (rank != FIRST_RANK) {
		prev_layer = (double*)malloc(sizeof(double) * LAYER_SIZE);
	}
	
	double* next_layer = NULL; // array to receive values from next process
	if (rank != LAST_RANK) {
		next_layer = (double*)malloc(sizeof(double) * LAYER_SIZE);
	}

	// Setting initial values of the function
	const double phi_0 = 0.0;

	double x, y, z;
	for (int i = 0; i < layers_count; ++i) {
		x = x0 + Hx * (layer_first + i);
		for (int j = 0; j < Ny; ++j) {
			y = y0 + Hy * j;
			for (int k = 0; k < Nz; ++k) {
				z = z0 + Hz * k;
				if (i == 0 || (i == layers_count - 1 && rank == LAST_RANK) || j == 0 || j == Ny - 1 || k == 0 || k == Nz - 1) {
					(*phi_values)[i * Ny * Nz + j * Nz + k] = Calculate_phi(x, y, z);
				} else {
					(*phi_values)[i * Ny * Nz + j * Nz + k] = phi_0;
				}
			}
		}
	}

	// Calculate phi_values while (max_delta_1 = max(i, j, k)|phi[m + 1] - phi[m]|) >= eps
	const int NO_TAG = 0;

	double max_delta_1;
	*iterations = 0;
	do {
		double max_delta_1_local = 0.0;

		MPI_Request request_send_prev, request_send_next;
		
		// send boundary layer to previous process
		if (rank != FIRST_RANK) {
			MPI_Isend(
				/* buf 		= */ *phi_values,
				/* count 	= */ LAYER_SIZE,
				/* datatype = */ MPI_DOUBLE,
				/* dest 	= */ rank - 1,
				/* tag 		= */ NO_TAG,
				/* comm 	= */ MPI_COMM_WORLD,
				/* request 	= */ &request_send_prev);
			MPI_Request_free(&request_send_prev);
		}

		// send boundary layer to next process
		if (rank != LAST_RANK) {
			MPI_Isend(
				/* buf 		= */ *phi_values + (layers_count - 1) * LAYER_SIZE,
				/* count 	= */ LAYER_SIZE,
				/* datatype = */ MPI_DOUBLE,
				/* dest 	= */ rank + 1,
				/* tag 		= */ NO_TAG,
				/* comm 	= */ MPI_COMM_WORLD,
				/* request 	= */ &request_send_next);
			MPI_Request_free(&request_send_next);
		}

		MPI_Request request_recv_prev, request_recv_next;

		// receive boundary layer from previous process
		if (rank != FIRST_RANK) {
			MPI_Irecv(
				/* buf 		= */ prev_layer,
				/* count 	= */ LAYER_SIZE,
				/* datatype = */ MPI_DOUBLE,
				/* source 	= */ rank - 1,
				/* tag 		= */ NO_TAG,
				/* comm 	= */ MPI_COMM_WORLD,
				/* request 	= */ &request_recv_prev);
		}

		// receive boundary layer from next process
		if (rank != LAST_RANK) {
			MPI_Irecv(
				/* buf 		= */ next_layer,
				/* count 	= */ LAYER_SIZE,
				/* datatype = */ MPI_DOUBLE,
				/* source 	= */ rank + 1,
				/* tag 		= */ NO_TAG,
				/* comm 	= */ MPI_COMM_WORLD,
				/* request 	= */ &request_recv_next);
		}		

		// Approximate phi_values from middle to bounds
		int middle_node_x = layer_first + layers_count / 2;
	
		if (middle_node_x != layer_first && middle_node_x != layer_last) {
			Approximate_layer(middle_node_x, layer_first, layer_last, Ny, Nz, 
				*phi_values, NULL, NULL, a, Hx, Hy, Hz, x0, y0, z0, &max_delta_1_local);
		}

		for (int i = 1; (middle_node_x - i > layer_first) || (middle_node_x + i < layer_last); ++i) {
			if (middle_node_x + i < layer_last) {
				Approximate_layer(middle_node_x + i, layer_first, layer_last, Ny, Nz, 
					*phi_values, NULL, NULL, a, Hx, Hy, Hz, x0, y0, z0, &max_delta_1_local);
			}
			if (middle_node_x - i > layer_first) {
				Approximate_layer(middle_node_x - i, layer_first, layer_last, Ny, Nz, 
					*phi_values, NULL, NULL, a, Hx, Hy, Hz, x0, y0, z0, &max_delta_1_local);
			}
		}

		// Wait for receive completion (from previous process)
		if (rank != FIRST_RANK) {
			MPI_Wait(&request_recv_prev, MPI_STATUS_IGNORE);
			if (layers_count != 1) {
				Approximate_layer(layer_first, layer_first, layer_last, Ny, Nz, 
					*phi_values, prev_layer, NULL, a, Hx, Hy, Hz, x0, y0, z0, &max_delta_1_local);
			}
		}

		// Wait for receive completion (from next process)
		if (rank != LAST_RANK) {
			MPI_Wait(&request_recv_next, MPI_STATUS_IGNORE);
			if (layers_count != 1) {
				Approximate_layer(layer_last, layer_first, layer_last, Ny, Nz, 
					*phi_values, NULL, next_layer, a, Hx, Hy, Hz, x0, y0, z0, &max_delta_1_local);
			}
		}
		
		// if only one layer on process approximate it with previous and next layers
		if (rank != LAST_RANK && rank != FIRST_RANK && layers_count == 1) {
			Approximate_layer(layer_last, layer_first, layer_last, Ny, Nz, 
					*phi_values, prev_layer, next_layer, a, Hx, Hy, Hz, x0, y0, z0, &max_delta_1_local);
		}

		MPI_Allreduce(
			/* sendbuf 	= */ &max_delta_1_local,
			/* recvbuf 	= */ &max_delta_1,
			/* count 	= */ 1,
			/* datatype = */ MPI_DOUBLE,
			/* op 		= */ MPI_MAX,
			/* comm 	= */ MPI_COMM_WORLD);

		(*iterations)++;
	} while (max_delta_1 >= eps);

	if (rank != FIRST_RANK) {
		free(prev_layer);
	}

	if (rank != LAST_RANK) {
		free(next_layer);
	}

	double max_delta_2_local = 0.0;
	double delta_iteration;

	for (int i = 0; i < layers_count; ++i) {
		x = x0 + (i + layer_first) * Hx;
		for (int j = 0; j < Ny; ++j) {
			y = y0 + j * Hy;
			for (int k = 0; k < Nz; ++k) {
				z = z0 + k * Hz;
				delta_iteration = (*phi_values)[i * Ny * Nz + j * Nz + k] - Calculate_phi(x, y, z);
				delta_iteration = (delta_iteration < 0) ? -delta_iteration : delta_iteration;

				if (delta_iteration > max_delta_2_local) {
					max_delta_2_local = delta_iteration;
				};
			}
		}
	}

	MPI_Allreduce(
			/* sendbuf 	= */ &max_delta_2_local,
			/* recvbuf 	= */ max_delta,
			/* count 	= */ 1,
			/* datatype = */ MPI_DOUBLE,
			/* op 		= */ MPI_MAX,
			/* comm 	= */ MPI_COMM_WORLD);
}