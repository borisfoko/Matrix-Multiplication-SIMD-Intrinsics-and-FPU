#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>


// Bericht über den Energieverbrauch
// DEBUG 
// Energy Consumption(mJ)
// 140,469.360
// 
// RELEASE
// Energy Consumption (mJ) 
// 59,315.186

#define MIN 10
#define MAX 200

float random_float(const float min, const float max)
{
	if (max == min) return min;
	else if (min < max) return (max - min) * ((float)rand() / RAND_MAX) + min;

	// return 0 if min > max
	return 0;
}

int multipy(float *MATRIX_A, float *MATRIX_B, float *MATRIX_C, int N) {
	int msec = 0;
	clock_t start, finish;
	start = clock();

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			// Multiply the row of MATRIX_A by the column of MATRIX_B to get the row of MATRIC_C.
			for (int k = 0; k < N; k++) {
				MATRIX_C[i * N + j] += MATRIX_A[i * N + k] * MATRIX_B[k * N + j];
			}
		}
	}

	finish = clock();
	msec = 1000.0 * (finish - start) / CLOCKS_PER_SEC;

	return msec;
}

void printMatrixF(float* MATRIX, int N) {
	for (int i = 0; i < N; i++) {
		printf("\n");
		for (int j = 0; j < N; j++) {
			printf("\t%f", MATRIX[i * N + j]);
		}
		printf("\n");
	}
}

void initMatrix(float* MATRIX, bool initToZero, int N) {
	if (initToZero) {
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				MATRIX[i * N + j] = 0;
			}
		}
	}
	else {
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				MATRIX[i * N + j] = random_float(MIN, MAX);
			}
		}
	}
}

// DEBUG
// Function/CallStack	CPU Time	Clockticks		Instructions Retired	CPI Rate	Retiring	Front-End Bound	Bad Speculation	Back-End Bound	Average CPU Frequency	Module				Function (Full)	Source File						Start Address
// multipy				7.174s		29,284,200,000	29,026,800, 000			1.009		25.1 %		0.3 %	  0.3 %					74.3 %			4.1 GHz 				mxm_standard.exe	multipy			MatrixMultiplicationStandard.c	0x411b90
// random_float			0.043s		162,000,000		156,600,000				1.034		33.3 %		16.7 %	  66.7 %				0.0 %			3.8 GHz 				mxm_standard.exe	random_float	MatrixMultiplicationStandard.c	0x411ea0
// initMatrix			0.005s		27,000,000		28,800,000				0.938		0.0 %		0.0 %	  100.0 %				0.0 %			5.8 GHz 				mxm_standard.exe	initMatrix		MatrixMultiplicationStandard.c	0x411860
// 
// RELEASE
// Function/CallStack	CPU Time	Clockticks		Instructions Retired	CPI Rate	Retiring	Front-End Bound	Bad Speculation	Back-End Bound	Average CPU Frequency	Module				Function (Full)	Source File						Start Address
// main					2.456s		10,481,400,000	8, 602, 200, 000		1.218		17.9 %		2.7 %	  1.4 %					78.0 %			4.3 GHz 				mxm_standard.exe	main			MatrixMultiplicationStandard.c	0x4010f01
// initMatrix			0.017s		57,600,000		99, 000, 000			0.582		0.0 %		0.0 %	  0.0 %					100.0 %			3.4 GHz 				mxm_standard.exe	initMatrix		MatrixMultiplicationStandard.c	0x401040
void main(void)
{
	int N = 1024;
	float* MATRIX_A = malloc(N * N * sizeof(float));
	float* MATRIX_B = malloc(N * N * sizeof(float));
	float* MATRIX_C = malloc(N * N * sizeof(float));

	//DATENSTRUKTUR MATRIX_A INITIALISIEREN
	initMatrix(MATRIX_A, false, N, N);

	//DATENSTRUKTUR MATRIX_A AUSGEBEN
	//printf("\n\n    Print out of matrix A\n\n");
	//printMatrixF(MATRIX_A, N, N);
	//printf("\n\n    End of A!\n\n");
	//_getch();


	//DATENSTRUKTUR MATRIX_B INITIALISIEREN
	//printf("\n\n    Input elements of matrix B\n\n");
	initMatrix(MATRIX_B, false, N);

	//DATENSTRUKTUR MATRIX_B AUSGEBEN
	//printf("\n\n    Print out of matrix B\n\n");
	//printMatrixF(MATRIX_B, N, N);
	//printf("\n\n    End of B!\n\n");
	//_getch();

	//DATENSTRUKTUR MATRIX_C INITIALISIEREN
	initMatrix(MATRIX_C, true, N);

	// Matrizenmultiplikation
	int standard_msec = multipy(MATRIX_A, MATRIX_B, MATRIX_C, N);
	
	//DATENSTRUKTUR MATRIX_C AUSGEBEN
	printf("\n\n    print out of matrix*matrix product C standard\n\n");
	printf("\n\n    Execution Time: %d in milliseconds\n\n", standard_msec);
	//printMatrixF(MATRIX_C, N);

	//_getch();
	free(MATRIX_A);
	free(MATRIX_B);
	free(MATRIX_C);
}