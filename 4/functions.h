#ifndef FUCTIONS_H
#define FUCTIONS_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void Read_arguments_values(
	int 	argc, 
	char* 	argv[], 
	int* 	Nx, 
	int* 	Ny, 
	int* 	Nz);

double Calculate_phi(
	double x, 
	double y, 
	double z);

double Calculate_ro(
	double a, 
	double x, 
	double y, 
	double z);

void Calculate_nodes(
	int rank, 
	int size, 
	int Nx, 
	int* count, 
	int* first, 
	int* last);

double Approximate_phi_value(
	double a, 
	double Hx, 
	double Hy, 
	double Hz, 
	double phi_nextX, 
	double phi_nextY, 
	double phi_nextZ, 
	double phi_prevX,
	double phi_prevY, 
	double phi_prevZ,
	double x, 
	double y, 
	double z);

void Approximate_layer(
	int layer_num,
	int layer_first,
	int layer_last,
	int Ny, 
	int Nz, 
	double* phi_values,
	const double* prev_layer, // significant only at layer_first
	const double* next_layer, // significant only at layer_last
	double a, 
	double Hx, 
	double Hy, 
	double Hz,
	double x0, 
	double y0, 
	double z0,
	double* max_delta);

void Use_Jacobi_method(
	int 	rank, 
	int 	size, 
	double 	a, 
	double 	eps, 
	double 	Dx, 
	double 	Dy, 
	double 	Dz,
	double 	x0,
	double	y0,
	double 	z0,
	int 	Nx, 
	int 	Ny, 
	int 	Nz,
	double** result,
	double* max_delta,
	int* iterations);

#endif