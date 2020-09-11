#include "functions.h"

void Read_input(int argc, char* argv[], int* grid_height, int* grid_width, 
	int* height1, int* width1, int* width2, 
	int* subheight1, int* subwidth1, int* subwidth2) {

	const int DIMS = 2;
	const int CONST_CNT = 3;

	if (argc != DIMS + CONST_CNT + 1) {
		MPI_Abort(MPI_COMM_WORLD, -1); // incorrect usage
	}

	*grid_height = atoi(argv[1]);
	*grid_width = atoi(argv[2]);
	*height1 = atoi(argv[3]);
	*width1 = atoi(argv[4]);
	*width2 = atoi(argv[5]);

	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	
	if (world_size != (*grid_height) * (*grid_width)) {
		MPI_Abort(MPI_COMM_WORLD, -2); // incorrect usage
	}

	if ((*height1) < 1 || (*width1) < 1 || (*width2) < 1 || (*grid_height) < 1 || (*grid_width) < 1) {
		MPI_Abort(MPI_COMM_WORLD, -3); // incorrect usage
	}

	if ((*height1) % (*grid_height) != 0 || (*width2) % (*grid_width) != 0) {
		MPI_Abort(MPI_COMM_WORLD, -4); // incorrect usage
	}

	*subheight1 = *height1 / *grid_height;
	*subwidth1 = *width1;
	*subwidth2 = *width2 / *grid_width;
}

void Create_grid_communicators(MPI_Comm comm_old, int height, int width, MPI_Comm* comm_row, MPI_Comm* comm_col) {

	const int DIMS = 2;
	const int TRUE_VAL = 1;

	int dims_grid[] = {height, width};
	int periods_grid[] = {0, 0};

	MPI_Comm comm_grid;

	MPI_Cart_create(
		/* comm_old 	= */ MPI_COMM_WORLD,
		/* ndims		= */ DIMS,
		/* dims 		= */ dims_grid,
		/* periods 		= */ periods_grid,
		/* reorder 		= */ TRUE_VAL,
		/* comm_cart	= */ &comm_grid);

	int remain_dims_row[] = {0, 1};
	MPI_Cart_sub(
		/* comm 		= */ comm_grid, 
		/* remain_dims 	= */ remain_dims_row, 
		/* newcomm 		= */ comm_row);

	int remain_dims_col[] = {1, 0};
	MPI_Cart_sub(
		/* comm 		= */ comm_grid, 
		/* remain_dims 	= */ remain_dims_col, 
		/* newcomm 		= */ comm_col);

	MPI_Comm_free(&comm_grid);
}

void Calculate_grid_ranks(MPI_Comm comm_row, MPI_Comm comm_col, 
	int* rank_row, int* rank_col, 
	int* rank_row_main, int* rank_col_main) {

	MPI_Comm_rank(comm_col, rank_col);
	MPI_Comm_rank(comm_row, rank_row);

	int coords1D_main[] = {0};

	MPI_Cart_rank(
		/* comm 	= */ comm_row,
		/* coords	= */ coords1D_main,
		/* rank		= */ rank_col_main);
	
	MPI_Cart_rank(
		/* comm 	= */ comm_col,
		/* coords	= */ coords1D_main,
		/* rank		= */ rank_row_main);
}

void Create_filled_matrix(int height, int width, double** matrix) {

	*matrix = (double*)malloc(sizeof(double) * height * width);
	for (int i = 0; i < height * width; ++i) {
		(*matrix)[i] = (i + 1) * (i + 1) / 3.;
	}
}

void Scatter_horizontal_stripes(const double* sendbuf, int sendheight, int sendwidth,
	double *recvbuf, int root, MPI_Comm comm) {

	int comm_size;
	MPI_Comm_size(comm, &comm_size);

	int count = sendheight / comm_size * sendwidth;

	MPI_Scatter(
			/* sendbuf 		= */ sendbuf, 
			/* sendcount 	= */ count,
			/* sendtype 	= */ MPI_DOUBLE, 
			/* recvbuf 		= */ recvbuf,
			/* recvcount 	= */ count,
			/* recvtype 	= */ MPI_DOUBLE,
			/* root 		= */ root,
			/* comm 		= */ comm);
}

void Scatter_vertical_stripes(const double* sendbuf, int sendheight, int sendwidth, 
	double *recvbuf, int root, MPI_Comm comm) {

	int comm_rank, comm_size;
	MPI_Comm_rank(comm, &comm_rank);
	MPI_Comm_size(comm, &comm_size);

	int recvwidth 	= sendwidth / comm_size;
	int recvheight 	= sendheight;

	MPI_Datatype recvtype_B, sendtype_B;

	// Type was made to receive matrix as 1 element
	MPI_Type_contiguous(
		/* count 		= */ recvwidth * recvheight,
		/* oldtype 		= */ MPI_DOUBLE,
		/* newtype 		= */ &recvtype_B);
	MPI_Type_commit(&recvtype_B);

	if (comm_rank == root) {
		MPI_Datatype tmptype;

		MPI_Type_vector(
			/* count 		= */ sendheight,
			/* blocklength 	= */ recvwidth,
			/* stride 		= */ sendwidth,
			/* oldtype 		= */ MPI_DOUBLE,
			/* newtype 		= */ &tmptype);
		MPI_Type_commit(&tmptype);

		MPI_Type_create_resized(
			/* oldtype		= */ tmptype,
			/* lb 			= */ 0,
			/* extent 		= */ sizeof(double) * recvwidth,
			/* newtype 		= */ &sendtype_B);
		MPI_Type_commit(&sendtype_B);

		MPI_Type_free(&tmptype);
	}

	MPI_Scatter(
		/* sendbuf 		= */ sendbuf, 
		/* sendcount 	= */ 1,
		/* sendtype 	= */ sendtype_B, 
		/* recvbuf 		= */ recvbuf,
		/* recvcount 	= */ 1,
		/* recvtype 	= */ recvtype_B,
		/* root 		= */ root,
		/* comm 		= */ comm);

	MPI_Type_free(&recvtype_B);

	if (comm_rank == root) {
		MPI_Type_free(&sendtype_B);
	}
}

void Multiplicate_matrices(int height1, int width1, int width2,
	const double* matrix1, const double* matrix2, double* matrixResult) {

	for (int i = 0; i < height1; ++i) {
		for (int j = 0; j < width2; ++j) {
			matrixResult[i * width2 + j] = 0;
			for (int k = 0; k < width1; ++k) {
				matrixResult[i * width2 + j] += matrix1[i * width1 + k] * matrix2[k * width2 + j];
			}
		}
	}
}

void Gather_result(const double* sendbuf, int sendheight, int sendwidth,
	double* recvbuf, int rank_col_main, int rank_row_main, MPI_Comm comm_col, MPI_Comm comm_row) {

	int rank_row, size_row;
	MPI_Comm_rank(comm_row, &rank_row);
	MPI_Comm_size(comm_row, &size_row);

	int rank_col, size_col;
	MPI_Comm_rank(comm_col, &rank_col);
	MPI_Comm_size(comm_col, &size_col);

	int recvheight 	= sendheight * size_col;
	int recvwidth 	= sendwidth * size_row;

	// Gather submatrices preC (m1 x n3) on processes in column #0
	double* preC = NULL;

	MPI_Datatype recvtype_subC;

	if (rank_row == rank_row_main) {
		preC = (double*)malloc(sizeof(double) * sendheight * recvwidth);

		MPI_Datatype tmptype;

		MPI_Type_vector(
			/* count 		= */ sendheight,
			/* blocklength 	= */ sendwidth,
			/* stride 		= */ recvwidth,
			/* oldtype 		= */ MPI_DOUBLE,
			/* newtype 		= */ &tmptype);
		MPI_Type_commit(&tmptype);

		MPI_Type_create_resized(
			/* oldtype		= */ tmptype,
			/* lb 			= */ 0,
			/* extent 		= */ sizeof(double) * sendwidth,
			/* newtype 		= */ &recvtype_subC);
		MPI_Type_commit(&recvtype_subC);

		MPI_Type_free(&tmptype);
	}

	MPI_Gather(
		/* sendbuf 		= */ sendbuf, 
		/* sendcount 	= */ sendheight * sendwidth, 
		/* sendtype 	= */ MPI_DOUBLE,
		/* recvbuf 		= */ preC, 
		/* recvcount 	= */ 1, 
		/* recvtype 	= */ recvtype_subC,
		/* root 		= */ rank_row_main, 
		/* comm 		= */ comm_row);
	
	// Gather matrix C (n1 x n3) on process with coordinates (0, 0) from column #0
	if (rank_row == rank_row_main) {
		MPI_Type_free(&recvtype_subC);

		MPI_Gather(
		/* sendbuf 		= */ preC, 
		/* sendcount 	= */ sendheight * recvwidth, 
		/* sendtype 	= */ MPI_DOUBLE,
		/* recvbuf 		= */ recvbuf, 
		/* recvcount 	= */ sendheight * recvwidth, 
		/* recvtype 	= */ MPI_DOUBLE,
		/* root 		= */ rank_col_main, 
		/* comm 		= */ comm_col);

		free(preC);
	}
}