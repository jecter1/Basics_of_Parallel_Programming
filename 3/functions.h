#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <mpi.h>
#include <stdlib.h>

void Read_input(
	int argc, 
	char* argv[], 
	int* grid_height, 
	int* grid_width, 
	int* height1, 
	int* width1, 
	int* width2, 
	int* subheight1, 
	int* subwidth1, 
	int* subwidth2);

void Create_grid_communicators(
	MPI_Comm comm_old, 
	int height, 
	int width, 
	MPI_Comm* comm_row, 
	MPI_Comm* comm_col);

void Calculate_grid_ranks(
	MPI_Comm comm_row, 
	MPI_Comm comm_col, 
	int* rank_row, 
	int* rank_col, 
	int* rank_row_main, 
	int* rank_col_main);

void Create_filled_matrix(
	int height, 
	int width,
	double** matrix);

void Scatter_horizontal_stripes(
	const double* sendbuf,
	int sendheight,
	int sendwidth,
	double *recvbuf,
	int root,
	MPI_Comm comm);

void Scatter_vertical_stripes(
	const double* sendbuf,
	int sendheight,
	int sendwidth,
	double *recvbuf,
	int root,
	MPI_Comm comm);

void Multiplicate_matrices(
	int height1, 
	int width1, 
	int width2, 
	const double* matrix1, 
	const double* matrix2, 
	double* matrixResult);

void Gather_result(
	const double* sendbuf,
	int sendheight,
	int sendwidth,
	double* recvbuf,
	int rank_col_main,
	int rank_row_main,
	MPI_Comm comm_col,
	MPI_Comm comm_row);

void Use_algorithm(int n1, int n2, int n3, int m1, int m2, int m3,
	int rank_col_main, int rank_row_main, int rank_col, int rank_row,
	MPI_Comm comm_col, MPI_Comm comm_row, double** C);

#endif