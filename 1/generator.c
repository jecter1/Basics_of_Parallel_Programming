#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

int isNumber(char* str, int len);

int main(int argc, char** argv) {
	if (argc != 2 || !isNumber(argv[1], strlen(argv[1]))) {
		fprintf(stderr, "Usage: %s size\n", argv[0]);
		return -1;
	} // incorrect usage

	srand(time(NULL));

	int N = atoi(argv[1]);

	//  MAKING MATRIX A
	double* A = (double*) malloc(sizeof(double) * N * N);
	if (A == NULL) {
		return -1;
	}

	const int min = -1000;
	const int max = 1000;
	for (int i = 0; i < N * N; ++i) {
		A[i] = ((double)rand() / RAND_MAX) * (max - min) + min;
	}

	double tmp;
	for (int j = 0; j < N; ++j) {
		for (int i = 0; i < N; ++i) {
			A[j * N + i] = (A[j * N + i] + A[i * N + j]) / 2;
			A[i * N + j] = A[j * N + i];
		}
		A[j * N + j] += N;
	}
	// ~MAKING MATRIX A~

	//  MAKING VECTOR b
	double* b = (double*) malloc(sizeof(double) * N);
	if (b == NULL) {
		free(A);
		return -1;
	}

	for (int i = 0; i < N; ++i) {
		b[i] = ((double)rand() / RAND_MAX) * (max - min) + min;
	}
	// ~MAKING VECTOR b~


	// open file
	FILE* fout = fopen("input.txt", "w+");
	if (fout == NULL) {
		perror("Opening 'input.txt'");
		free(A);
		free(b);
		return -1;
	}

	// print N
	if (fprintf(fout, "%d\n", N) < 0) {
		if (fclose(fout)) {
			perror("Closing 'input.txt'");
		}
		perror("Printing matrix");
		free(A);
		free(b);
		return -1;
	}

	// print A
	for (int i = 0; i < N * N; ++i) {
		if (fprintf(fout, "%lf ", A[i]) < 0) {
			if (fclose(fout)) {
				perror("Closing 'input.txt'");
			}
			perror("Printing matrix");
			free(A);
			free(b);
			return -1;
		}
	}
	free(A);

	// print '\n'
	if (fprintf(fout, "\n") < 0) {
		if (fclose(fout)) {
			perror("Closing 'input.txt'");
		}
		perror("fprintf");
		free(b);
		return -1;
	}

	// print b
	for (int i = 0; i < N; ++i) {
		if (fprintf(fout, "%lf ", b[i]) < 0) {
			if (fclose(fout)) {
				perror("Closing 'input.txt'");
			}
			perror("Printing vector");
			free(b);
			return -1;
		}
	}
	free(b);

	// print '\n'
	if (fprintf(fout, "\n") < 0) {
		if (fclose(fout)) {
			perror("Closing 'input.txt'");
		}
		perror("fprintf");
		return -1;
	}

	// close file
	if (fclose(fout) != 0) {
		fprintf(stderr, "Closing 'input.txt'");
		return -1;
	}
	return 0;
}

int isNumber(char* str, int len) {
	if (str == NULL) {
		fprintf(stderr, "isNumber(char* str, int len). str is NULL!\n");
		exit(1);
	} // incorrect usage
	if (len < 0) {
		fprintf(stderr, "isNumber(char* str, int len). len < 0!\n");
		exit(1);
	} // incorrect usage

	if (!(str[0] >= '0' && str[0] <= '9') && str[0] != '-' && str[0] != '+') {
		return 0;
	} // examlpe: "a..."

	if ((str[0] == '-' || str[0] == '+') && len == 1) {
		return 0;
	} // example: "-"

	for (int i = 1; (i < len) && isNumber; i++) {
		if (!(str[i] >= '0' && str[i] <= '9')) {
			return 0;
		}
	} // example: "-124a..."

	return 1;
}