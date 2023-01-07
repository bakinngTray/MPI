#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"
#include <iostream>

#define N 1500

using namespace std;


void matrixes_init(double*& A, double*& B, double*& C, double*& result) {

	A = new double[N * N];
	B = new double[N * N];
	C = new double[N * N];
	result = new double[N * N];

	for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) A[i * N + j] = B[i * N + j] = 2;
	for (int i = 0; i < N * N; i++) C[i] = result[i] = 0;
}


void mul_matrix(double* A, double* B, double* C, int mod_type, int mod) {
	int i, j, k;
	for (i = 0; i < N; i++) {
		for (j = 0 + mod_type; j < N; j += mod) {
			for (k = 0; k < N; k++) {
				C[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
}


void main_process(int comm_size, int comm_runk, double* A, double* B, double* C, double* result) {
	cout << "run..." << endl;
	double t1 = MPI_Wtime();
	for (int nCounter = 1; nCounter < comm_size; nCounter++) {
		MPI_Send(A, N * N, MPI_DOUBLE, nCounter, 1, MPI_COMM_WORLD);
		MPI_Send(B, N * N, MPI_DOUBLE, nCounter, 2, MPI_COMM_WORLD);
		MPI_Send(C, N * N, MPI_DOUBLE, nCounter, 3, MPI_COMM_WORLD);
	}
	mul_matrix(A, B, C, comm_runk, comm_size);
	MPI_Reduce(C, result, N * N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	cout << "total time - " << MPI_Wtime() - t1 << endl;
}


void subprocess(int comm_runk, int comm_size, double*& A, double*& B, double*& C, double* result) {
	A = new double[N * N];
	B = new double[N * N];
	C = new double[N * N];
	MPI_Recv(A, N * N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(B, N * N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(C, N * N, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	mul_matrix(A, B, C, comm_runk, comm_size);
	MPI_Reduce(C, result, N * N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
}


int main(int argc, char* argv[]) {
	double* A, * B, * C;
	double* result = 0;
	int comm_runk, comm_size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &comm_runk);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	if (comm_runk == 0) {
		matrixes_init(A, B, C, result);
		main_process(comm_size, comm_runk, A, B, C, result);
	}
	else subprocess(comm_runk, comm_size, A, B, C, result);
	MPI_Finalize();
}