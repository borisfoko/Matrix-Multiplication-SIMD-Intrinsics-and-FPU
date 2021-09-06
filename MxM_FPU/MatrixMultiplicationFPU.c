#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

// Bericht über den Energieverbrauch
// DEBUG 
// Energy Consumption(mJ)
// 130,589.722
// 
// RELEASE
// Energy Consumption (mJ) 
// 2,218.933

#define MIN 10
#define MAX 200

// Offset for mat[i][j], w is an row width, t == 1 for transposed access.
#define mi(w, t, i, j) 4 * ((i * w + j) * (1-t) + (j * w + i) * t)

// Load & multiply.
#define flm(k, i, j, m, n, a, b) \
	__asm fld dword ptr [ebx + mi(m, a, i, k)] \
	__asm fmul dword ptr [ecx + mi(n, b, k, j)]

// Implementation for 6x6 Matrix
#define e6(i, j, l, m, n, a, b) \
	flm(0, i, j, m, n, a, b) \
	flm(1, i, j, m, n, a, b) \
	flm(2, i, j, m, n, a, b) \
	flm(3, i, j, m, n, a, b) \
	flm(4, i, j, m, n, a, b) \
	flm(5, i, j, m, n, a, b) \
	__asm faddp st(1), st(0) \
	__asm fxch st(2) \
	__asm faddp st(1), st(0) \
	__asm faddp st(1), st(0) \
	__asm fxch st(2) \
	__asm faddp st(1), st(0) \
	__asm faddp st(1), st(0) \
	__asm fstp dword ptr [eax + mi(l, 0, i, j)]


/*
By default start first loop			      ;	for (int i = 0; i < N; i++) {
LN1 First for loop increment counter      ;	   i = i + 1
LN2 First for loop compare i with N       ;    i < N
After LN2 start the second loop			  ;	   for (int j = 0; j < N; j++) {
LN3 Second for loop increment counter     ;			j = j + 1
LN4 Second for loop compare i with N      ;			j < N
After LN4 start the third loop			  ;			for (int inner = 0; inner < N; inner++) {
LN5 Third for loop increment counter      ;				inner = inner + 1
LN6 Multiplication and Addition			  ;             MATRIX_C[i * N + j] += MATRIX_A[i * N + inner] * MATRIX_B[inner * N + j];
LN7 Close third for loop				  ;			}
LN8 Close second for loop				  ;		}
LN9 Close first for loop				  ;	}
LN10, LN11 Clean and return
*/
int multiply_fpu(float* MATRIX_A, float* MATRIX_B, float* MATRIX_C, int N) {
	int msec = 0;
	clock_t start, finish;
	start = clock();

	__asm {
		push        ebp
		sub         esp, 0E4h
		push        ebx
		push        esi
		push        edi
		lea         edi, [ebp - 0E4h]
		mov         ecx, 39h
		mov         eax, 0CCCCCCCCh
		rep stos    dword ptr es : [edi]
		mov         dword ptr[ebp - 8], 0
		jmp LN2
		LN1 :
			mov eax, dword ptr[ebp - 8]
			add eax, 1
			mov dword ptr[ebp - 8], eax
		LN2 :
			mov         eax, dword ptr[ebp - 8]
			cmp         eax, dword ptr[N]
			jge LN10
			mov         dword ptr[ebp - 14h], 0
			jmp LN4
		LN3 :
			mov         eax, dword ptr[ebp - 14h]
			add         eax, 1
			mov         dword ptr[ebp - 14h], eax
		LN4 :
			mov         eax, dword ptr[ebp - 14h]
			cmp         eax, dword ptr[N]
			jge LN9
			mov         dword ptr[ebp - 20h], 0
			jmp         LN6
		LN5 :
			mov         eax, dword ptr[ebp - 20h]
			add         eax, 1
			mov         dword ptr[ebp - 20h], eax
		LN6 :
			mov         eax, dword ptr[ebp - 20h]
			cmp         eax, dword ptr[N]
			jge         LN8
			mov         eax, dword ptr[ebp - 8]
			imul        eax, dword ptr[N]
			add         eax, dword ptr[ebp - 14h]
			mov         ecx, dword ptr[ebp - 8]
			imul        ecx, dword ptr[N]
			add         ecx, dword ptr[ebp - 20h]
			mov         edx, dword ptr[ebp - 20h]
			imul        edx, dword ptr[N]
			add         edx, dword ptr[ebp - 14h]
			mov         esi, dword ptr[MATRIX_A]
			mov         edi, dword ptr[MATRIX_B]
			fld			dword ptr[esi + ecx * 4]
			fmul        dword ptr[edi + edx * 4]
			mov         ecx, dword ptr[MATRIX_C]
			fadd        dword ptr[ecx + eax * 4]
			mov         edx, dword ptr[ebp - 8]
			imul        edx, dword ptr[N]
			add         edx, dword ptr[ebp - 14h]
			mov         eax, dword ptr[MATRIX_C]
			fstp        dword ptr[eax + edx * 4]
		LN7 :
			jmp LN5
		LN8 :
			jmp LN3
		LN9 :
			jmp LN1
		LN10 :
			pop         edi
		LN11 :
			pop esi
			pop ebx
			add         esp, 0E4h
			cmp         ebp, esp
			mov         esp, ebp
			pop ebp
			ret
	}

	finish = clock();
	msec = 1000.0 * (finish - start) / CLOCKS_PER_SEC;

	return msec;
}

int multiply_6x6(float* MATRIX_A, float* MATRIX_B, float* MATRIX_C) {
	int msec = 0;
	clock_t start, finish;
	start = clock();

	__asm mov ebx, DWORD PTR[MATRIX_A]
	__asm mov ecx, DWORD PTR[MATRIX_B]
	__asm mov eax, DWORD PTR[MATRIX_C]
	e6(0, 0, 6, 6, 6, 0, 0)
	e6(0, 1, 6, 6, 6, 0, 0)
	e6(0, 2, 6, 6, 6, 0, 0)
	e6(0, 3, 6, 6, 6, 0, 0)
	e6(0, 4, 6, 6, 6, 0, 0)
	e6(0, 5, 6, 6, 6, 0, 0)
	e6(1, 0, 6, 6, 6, 0, 0)
	e6(1, 1, 6, 6, 6, 0, 0)
	e6(1, 2, 6, 6, 6, 0, 0)
	e6(1, 3, 6, 6, 6, 0, 0)
	e6(1, 4, 6, 6, 6, 0, 0)
	e6(1, 5, 6, 6, 6, 0, 0)
	e6(2, 0, 6, 6, 6, 0, 0)
	e6(2, 1, 6, 6, 6, 0, 0)
	e6(2, 2, 6, 6, 6, 0, 0)
	e6(2, 3, 6, 6, 6, 0, 0)
	e6(2, 4, 6, 6, 6, 0, 0)
	e6(2, 5, 6, 6, 6, 0, 0)
	e6(3, 0, 6, 6, 6, 0, 0)
	e6(3, 1, 6, 6, 6, 0, 0)
	e6(3, 2, 6, 6, 6, 0, 0)
	e6(3, 3, 6, 6, 6, 0, 0)
	e6(3, 4, 6, 6, 6, 0, 0)
	e6(3, 5, 6, 6, 6, 0, 0)
	e6(4, 0, 6, 6, 6, 0, 0)
	e6(4, 1, 6, 6, 6, 0, 0)
	e6(4, 2, 6, 6, 6, 0, 0)
	e6(4, 3, 6, 6, 6, 0, 0)
	e6(4, 4, 6, 6, 6, 0, 0)
	e6(4, 5, 6, 6, 6, 0, 0)
	e6(5, 0, 6, 6, 6, 0, 0)
	e6(5, 1, 6, 6, 6, 0, 0)
	e6(5, 2, 6, 6, 6, 0, 0)
	e6(5, 3, 6, 6, 6, 0, 0)
	e6(5, 4, 6, 6, 6, 0, 0)
	e6(5, 5, 6, 6, 6, 0, 0)

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

float random_float(const float min, const float max)
{
	if (max == min) return min;
	else if (min < max) return (max - min) * ((float)rand() / RAND_MAX) + min;

	// return 0 if min > max
	return 0;
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
// Function/CallStack	CPU Time	Clockticks		Instructions Retired	CPI Rate	Retiring	Front-End Bound	Bad Speculation	Back-End Bound	Average CPU Frequency	Module						Function (Full)	Source File		Start Address
// multiply_fpu			6.906s		27,495,000,000	29,035,800,000			0.947		26.5 %		0.2 %	  0.6 %					72.7 %			4.0 GHz 				mxm_fpu.exe	multiply_fpu	MatrixMultiplicationFPU.c		0x412810
// random_float			0.042s		162,000,000		172,800,000				0.938		0.0 %		0.0 %	  0.0 %					100.0 %			3.8 GHz 				mxm_fpu.exe	random_double	MatrixMultiplicationFPU.c		0x412cc0
// initMatrix			0.004s		27,000,000		34,200,000				0.789		0.0 %		0.0 %	  0.0 %					100.0 %			6.9 GHz 				mxm_fpu.exe	initMatrix		MatrixMultiplicationFPU.c		0x411880


// RELEASE
// Function/CallStack	CPU Time	Clockticks		Instructions Retired	CPI Rate	Retiring	Front-End Bound	Bad Speculation	Back-End Bound	Average CPU Frequency	Module				Function (Full)	Source File						Start Address
// initMatrix			13.281ms	46, 800, 000	104, 400, 000			0.448		0.0 %		0.0 %	  0.0 %					100.0 %			3.5 GHz 				mxm_fpu.exe			initMatrix		MatrixMultiplicationFPU.c		0x401150

// All oder functions where not recorded in release mode because of optimization. Have a look at MxM_Streaming_Data

void main(void)
{
	int N = 1024;
	float* MATRIX_A = malloc(N * N * sizeof(float));
	float* MATRIX_B = malloc(N * N * sizeof(float));
	float* MATRIX_C = malloc(N * N * sizeof(float));

	//DATENSTRUKTUR MATRIX_A INITIALISIEREN
	//printf("\n\n    Input elements of matrix A\n\n");
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
	//int standard_msec = multipy(MATRIX_A, MATRIX_B, MATRIX_C, N);

	//DATENSTRUKTUR MATRIX_C AUSGEBEN
	//printf("\n\n    print out of matrix*matrix product C standard\n\n");
	//printf("\n\n    Execution Time: %d in milliseconds\n\n", standard_msec);
	//printMatrixF(MATRIX_C, N);

	//DATENSTRUKTUR MATRIX_C INITIALISIEREN
	//initMatrix(MATRIX_C, true, N);

	// Matrizenmultiplikation with asm
	//int asm_msec = multiply_asm(MATRIX_A, MATRIX_B, MATRIX_C, N);

	//DATENSTRUKTUR MATRIX_C AUSGEBEN
	//printf("\n\n    print out of matrix*matrix product C with asm\n\n");
	//printf("\n\n    Execution Time: %d in milliseconds\n\n", asm_msec);
	//printMatrixF(MATRIX_C, N);

	//DATENSTRUKTUR MATRIX_C INITIALISIEREN
	//initMatrix(MATRIX_C, true, N);

	// Matrizenmultiplikation with fpu
	int fpu_msec = multiply_fpu(MATRIX_A, MATRIX_B, MATRIX_C, N);

	//DATENSTRUKTUR MATRIX_C AUSGEBEN
	printf("\n\n    print out of matrix*matrix product C with fpu\n\n");
	printf("\n\n    Execution Time: %d in milliseconds\n\n", fpu_msec);
	//printMatrixF(MATRIX_C, N);

	//_getch();
	free(MATRIX_A);
	free(MATRIX_B);
	free(MATRIX_C);
}