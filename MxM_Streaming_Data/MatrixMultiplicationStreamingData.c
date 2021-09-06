#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <immintrin.h>

#define MIN 10
#define MAX 200

int multipy(const double* MATRIX_A, const double* MATRIX_B, double* MATRIX_C, int N) {
	int msec = 0;
	clock_t start, finish;
	start = clock();

	for (int i = 0; i < N; i++) {
		// Clockticks	Instructions Retired	CPI Rate	Retiring	Front - End Bound	Bad Speculation		Back - End Bound
		// 5,400,000	5,400,000				1.000		0.0 %		0.0 %				0.0 %				0.0 %
		for (int j = 0; j < N; j++) {
			// Multiply the row of MATRIX_A by the column of MATRIX_B to get the row of MATRIC_C.
			// Clockticks		Instructions Retired	CPI Rate	Retiring	Front - End Bound	Bad Speculation		Back - End Bound
			// 1,112,400,000	835,200,000				1.332		0.5 %		0.0 %				0.0 %				8.8 %
			for (int k = 0; k < N; k++) {
				// Clockticks	    Instructions Retired	CPI Rate	Retiring	Front-End Bound	Bad Speculation	Back-End Bound
				// 47,608,200,000	28,209,600,000			1.688		14.2 %		0.2 %			0.8 %			80.3 %
				MATRIX_C[i * N + j] += MATRIX_A[i * N + k] * MATRIX_B[k * N + j];
			}
		}
	}

	finish = clock();
	msec = 1000.0 * (finish - start) / CLOCKS_PER_SEC;

	return msec;
}

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

//MMX technology cannot handle single precision float, so we will use short(int16) and in32 as result
int multipy_mmx(const short* MATRIX_A, const short* MATRIX_B, int* MATRIX_C, int N)
{
	int msec = 0;
	clock_t start, finish;
	start = clock();

	__m64 a_line,  b_line;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			__m64 sum = _mm_setzero_si64();		// init sum to zero 
			int sum_down = 0;
			for (int k = 0; k < N; k+=4) {      // add 4 short at the same time using the MMX _m_pmaddwd function equivalent to asm pmaddwd 
				a_line = _mm_set_pi16(MATRIX_A[i * N + k], MATRIX_A[i * N + k + 1], MATRIX_A[i * N + k + 2], MATRIX_A[i * N + k + 3]);
				b_line = _mm_set_pi16(MATRIX_B[k * N + j], MATRIX_B[(k + 1) * N + j], MATRIX_B[(k + 2) * N + j], MATRIX_B[(k + 3) * N + j]);

				sum = _m_paddw(sum, _m_pmaddwd(a_line, b_line));
			}
			sum_down = _mm_cvtsi64_si32(sum);   // save low 32 bits
			sum = _m_psrlqi(sum, 32);			// shift right on 32 bits
			sum_down += _mm_cvtsi64_si32(sum);  // save low 32 bits
			MATRIX_C[i * N + j] = sum_down;
		}
	}

	// Clear the MMX registers and MMX state
	_m_empty();
	_mm_empty();

	finish = clock();
	msec = 1000.0 * (finish - start) / CLOCKS_PER_SEC;

	return msec;
}

int multipy_sse(const float* MATRIX_A, const float* MATRIX_B, float* MATRIX_C, int N)
{
	int msec = 0;
	clock_t start, finish;
	start = clock();

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j += 4) {
			__m128 sum = _mm_load_ps(MATRIX_C + i * N + j);
			for (int k = 0; k < N; k ++) {
				sum = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(MATRIX_A[i * N + k]), _mm_load_ps(MATRIX_B + k * N + j)), sum);
			}
			_mm_store_ps(MATRIX_C + i * N + j, sum);
		}
	}

	finish = clock();
	msec = 1000.0 * (finish - start) / CLOCKS_PER_SEC;

	return msec;
}

int multipy_sse2(const double* MATRIX_A, const double* MATRIX_B, double* MATRIX_C, int N)
{
	int msec = 0;
	clock_t start, finish;
	start = clock();

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j += 2) {
			__m128d sum = _mm_load_pd(MATRIX_C + i * N + j);
			for (int k = 0; k < N; k++) {
				sum = _mm_add_pd(_mm_mul_pd(_mm_set_pd1(MATRIX_A[i * N + k]), _mm_load_pd(MATRIX_B + k * N + j)), sum);
			}
			_mm_store_pd(MATRIX_C + i * N + j, sum);
		}
	}

	finish = clock();
	msec = 1000.0 * (finish - start) / CLOCKS_PER_SEC;

	return msec;
}

int multipy_avx(const float* MATRIX_A, const float* MATRIX_B, float* MATRIX_C, int N)
{
	int msec = 0;
	clock_t start, finish;
	start = clock();

	const int block_width = N >= 256 ? 512 : 256;
	const int block_height = N >= 512 ? 8 : N >= 256 ? 16 : 32;
	for (int row_offset = 0; row_offset < N; row_offset += block_height) {
		for (int column_offset = 0; column_offset < N; column_offset += block_width) {
			for (int i = 0; i < N; ++i) {
				for (int j = column_offset; j < column_offset + block_width && j < N; j += 8) {
					__m256 sum = _mm256_load_ps(MATRIX_C + i * N + j);
					for (int k = row_offset; k < row_offset + block_height && k < N; ++k) {
						sum = _mm256_fmadd_ps(_mm256_set1_ps(MATRIX_A[i * N + k]), _mm256_load_ps(MATRIX_B + k * N + j), sum);
					}
					_mm256_store_ps(MATRIX_C + i * N + j, sum);
				}
			}
		}
	}

	finish = clock();
	msec = 1000.0 * (finish - start) / CLOCKS_PER_SEC;

	return msec;
}

float random_double(const double min, const double max)
{
	if (max == min) return min;
	else if (min < max) return (max - min) * ((double)rand() / RAND_MAX) + min;

	return 0;
}

void printMatrixI(int* MATRIX, int N) {
	for (int i = 0; i < N; i++) {
		printf("\n");
		for (int j = 0; j < N; j++) {
			printf("\t%d", MATRIX[i * N + j]);
		}
		printf("\n");
	}
}

void printMatrixS(const short* MATRIX, int N) {
	for (int i = 0; i < N; i++) {
		printf("\n");
		for (int j = 0; j < N; j++) {
			printf("\t%d", MATRIX[i * N + j]);
		}
		printf("\n");
	}
}

void printMatrixF(const float* MATRIX, int N) {
	for (int i = 0; i < N; i++) {
		printf("\n");
		for (int j = 0; j < N; j++) {
			printf("\t%f", MATRIX[i * N + j]);
		}
		printf("\n");
	}
}

void printMatrixD(const double* MATRIX, int N) {
	for (int i = 0; i < N; i++) {
		printf("\n");
		for (int j = 0; j < N; j++) {
			printf("\t%lf", MATRIX[i * N + j]);
		}
		printf("\n");
	}
}

void initMatrixD(double* MATRIX, bool initToZero, int N) {
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
				MATRIX[i * N + j] = random_double(MIN, MAX);
			}
		}
	}
}

void initMatrixF(float* MATRIX, bool initToZero, int N) {
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
				MATRIX[i * N + j] = (float) random_double(MIN, MAX);
			}
		}
	}
}

void copyMatrixDF(const double* MATRIX_SOURCE, float* MATRIX_TARGET, int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			MATRIX_TARGET[i * N + j] = (float) MATRIX_SOURCE[i * N + j];
		}
	}
}

void copyMatrixDI(const double* MATRIX_SOURCE, int* MATRIX_TARGET, int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			MATRIX_TARGET[i * N + j] = (int) MATRIX_SOURCE[i * N + j];
		}
	}
}

void copyMatrixDS(const double* MATRIX_SOURCE, short* MATRIX_TARGET, int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			MATRIX_TARGET[i * N + j] = (short) MATRIX_SOURCE[i * N + j];
		}
	}
}

void runAndSaveMxM(int maxN) {
	// Define all needed MATRIX(s)
	double *MATRIX_D_A, *MATRIX_D_B, *MATRIX_D_C;
	float *MATRIX_F_A, *MATRIX_F_B, *MATRIX_F_C;
	short *MATRIX_S_A, *MATRIX_S_B;
	int* MATRIX_I_C;

	// Define time saver
	int standard_msec, fpu_msec, mmx_msec, sse_msec, sse2_msec, avx_msec;

	// Define and create file
	FILE* fpt;
	fpt = fopen("statistic.csv", "w+");

	// Write header
	fprintf(fpt, "Matrix Size, Standard, FPU, MMX, SSE, SSE2, AVX\n");
	int NOld = 0;
	int NIncrement = 8;
	for (int N = 8; N <= maxN; N += NIncrement) {
		if ((N - NOld) >= 208) {
			NOld = N;
			NIncrement = NIncrement + 8;
		}
		printf("\nMatrix Size: %dx%d", N, N);
		MATRIX_D_A = malloc(N * N * sizeof(double));
		MATRIX_D_B = malloc(N * N * sizeof(double));
		MATRIX_D_C = malloc(N * N * sizeof(double));
		
		MATRIX_F_A = malloc(N * N * sizeof(float));
		MATRIX_F_B = malloc(N * N * sizeof(float));
		MATRIX_F_C = malloc(N * N * sizeof(float));

		MATRIX_S_A = malloc(N * N * sizeof(short));
		MATRIX_S_B = malloc(N * N * sizeof(short));
		MATRIX_I_C = malloc(N * N * sizeof(int));
		
		// Init MATRIX_A
		initMatrixD(MATRIX_D_A, false, N);

		// Copy MATRIX_D_A to MATRIX_F_A
		copyMatrixDF(MATRIX_D_A, MATRIX_F_A, N);

		// Copy MATRIX_D_A to MATRIX_S_A
		copyMatrixDS(MATRIX_D_A, MATRIX_S_A, N);

		// Init MATRIX_B
		initMatrixD(MATRIX_D_B, false, N);

		// Copy MATRIX_D_A to MATRIX_F_A
		copyMatrixDF(MATRIX_D_B, MATRIX_F_B, N);

		// Copy MATRIX_D_B to MATRIX_S_B
		copyMatrixDS(MATRIX_D_B, MATRIX_S_B, N);

		// Init MATRIX_C
		initMatrixD(MATRIX_D_C, true, N);

		// Copy MATRIX_D_C to MATRIX_F_C
		copyMatrixDF(MATRIX_D_C, MATRIX_F_C, N);

		// Copy MATRIX_D_C to MATRIX_I_C
		copyMatrixDI(MATRIX_D_C, MATRIX_I_C, N);

		// Matrix multiplication standard
		standard_msec = multipy(MATRIX_D_A, MATRIX_D_B, MATRIX_D_C, N);

		// Matrix multiplicatio with fpu
		fpu_msec = multiply_fpu(MATRIX_F_A, MATRIX_F_B, MATRIX_F_C, N);

		// Matrix multiplication MMX
		mmx_msec = multipy_mmx(MATRIX_S_A, MATRIX_S_B, MATRIX_I_C, N);

		// Init MATRIX_C
		initMatrixF(MATRIX_F_C, true, N);

		// Matrix multiplication SSE
		sse_msec = multipy_sse(MATRIX_F_A, MATRIX_F_B, MATRIX_F_C, N);

		// Init MATRIX_C
		initMatrixD(MATRIX_D_C, true, N);

		// Matrix multiplication SSE2
		sse2_msec = multipy_sse2(MATRIX_D_A, MATRIX_D_B, MATRIX_D_C, N);

		// Init MATRIX_C
		initMatrixF(MATRIX_F_C, true, N);

		// Matrix multiplication AVX
		avx_msec = multipy_avx(MATRIX_F_A, MATRIX_F_B, MATRIX_F_C, N);

		fprintf(fpt, "%d, %d, %d, %d, %d, %d, %d\n", N, standard_msec, fpu_msec, mmx_msec, sse_msec, sse2_msec, avx_msec);

		// Free MATRIX Memory
		free(MATRIX_S_A);
		free(MATRIX_S_B);
		free(MATRIX_I_C);
		free(MATRIX_D_A);
		free(MATRIX_D_B);
		free(MATRIX_D_C);
		free(MATRIX_F_A);
		free(MATRIX_F_B);
		free(MATRIX_F_C);
	}

	// Close file
	fclose(fpt);

	printf("\nExecution terminated");
}

void singleRun(int N) {
	// Define all needed MATRIX(s)
	double* MATRIX_D_A, * MATRIX_D_B, * MATRIX_D_C;
	float* MATRIX_F_A, * MATRIX_F_B, * MATRIX_F_C;
	short* MATRIX_S_A, * MATRIX_S_B;
	int* MATRIX_I_C;

	printf("\nMatrix Size: %dx%d", N, N);
	MATRIX_D_A = malloc(N * N * sizeof(double));
	MATRIX_D_B = malloc(N * N * sizeof(double));
	MATRIX_D_C = malloc(N * N * sizeof(double));

	MATRIX_F_A = malloc(N * N * sizeof(float));
	MATRIX_F_B = malloc(N * N * sizeof(float));
	MATRIX_F_C = malloc(N * N * sizeof(float));

	MATRIX_S_A = malloc(N * N * sizeof(short));
	MATRIX_S_B = malloc(N * N * sizeof(short));
	MATRIX_I_C = malloc(N * N * sizeof(int));

	// Init MATRIX_A
	initMatrixD(MATRIX_D_A, false, N);

	// Copy MATRIX_D_A to MATRIX_F_A
	copyMatrixDF(MATRIX_D_A, MATRIX_F_A, N);

	// Copy MATRIX_D_A to MATRIX_S_A
	copyMatrixDS(MATRIX_D_A, MATRIX_S_A, N);

	// Init MATRIX_B
	initMatrixD(MATRIX_D_B, false, N);

	// Copy MATRIX_D_A to MATRIX_F_A
	copyMatrixDF(MATRIX_D_B, MATRIX_F_B, N);

	// Copy MATRIX_D_B to MATRIX_S_B
	copyMatrixDS(MATRIX_D_B, MATRIX_S_B, N);

	// Init MATRIX_C
	initMatrixD(MATRIX_D_C, true, N);

	// Copy MATRIX_D_C to MATRIX_F_C
	copyMatrixDF(MATRIX_D_C, MATRIX_F_C, N);

	// Copy MATRIX_D_C to MATRIX_I_C
	copyMatrixDI(MATRIX_D_C, MATRIX_I_C, N);

	// Matrix multiplication standard
	int standard_msec = multipy(MATRIX_D_A, MATRIX_D_B, MATRIX_D_C, N);

	// Matrix multiplicatio with fpu
	int fpu_msec = multiply_fpu(MATRIX_F_A, MATRIX_F_B, MATRIX_F_C, N);

	// Matrix multiplication MMX
	int mmx_msec = multipy_mmx(MATRIX_S_A, MATRIX_S_B, MATRIX_I_C, N);

	// Init MATRIX_C
	initMatrixF(MATRIX_F_C, true, N);

	// Matrix multiplication SSE
	int sse_msec = multipy_sse(MATRIX_F_A, MATRIX_F_B, MATRIX_F_C, N);

	// Init MATRIX_C
	initMatrixD(MATRIX_D_C, true, N);

	// Matrix multiplication SSE2
	int sse2_msec = multipy_sse2(MATRIX_D_A, MATRIX_D_B, MATRIX_D_C, N);

	// Init MATRIX_C
	initMatrixF(MATRIX_F_C, true, N);

	// Matrix multiplication AVX
	int avx_msec = multipy_avx(MATRIX_F_A, MATRIX_F_B, MATRIX_F_C, N);

	printf("\nMatrix Size: %d, Standard: %d, FPU: %d, MMX: %d, SSE: %d, SSE2: %d, AVX: %d\n", N, standard_msec, fpu_msec, mmx_msec, sse_msec, sse2_msec, avx_msec);

	// Free MATRIX Memory
	free(MATRIX_S_A);
	free(MATRIX_S_B);
	free(MATRIX_I_C);
	free(MATRIX_D_A);
	free(MATRIX_D_B);
	free(MATRIX_D_C);
	free(MATRIX_F_A);
	free(MATRIX_F_B);
	free(MATRIX_F_C);
}

// DEBUG
// Function/CallStack	CPU Time	Clockticks		Instructions Retired	CPI Rate	Retiring	Front-End Bound	Bad Speculation	Back-End Bound	Average CPU Frequency	Module					Function (Full)		Source File							Start Address
// multipy				14.600s		48,727,800,000	29, 050, 200, 000		1.677		14.7 %		0.2 %	  0.8 %					84.4 %			3.3 GHz 				mxm_streaming_data.exe	multipy				MatrixMultiplicationStreamingData.c	0x412080
// multiply_fpu			6.874s		27,883,800,000	29, 030, 400, 000		0.961		28.3 %		0.5 %     0.0 %					72.1 %			4.1 GHz 				mxm_streaming_data.exe	multiply_fpu		MatrixMultiplicationStreamingData.c	0x411ee0
// multipy_sse2			5.267s		17,299,800,000	12, 385, 800, 000		1.397		19.0 %		0.2 %	  0.2 %					80.7 %			3.3 GHz 				mxm_streaming_data.exe	multipy_sse2		MatrixMultiplicationStreamingData.c	0x412980
// multipy_mmx			5.000s		18,208,800,000	18, 304, 200, 000		0.995		24.3 %		0.2 %     1.3 %					74.2 %			3.6 GHz 				mxm_streaming_data.exe	multipy_mmx			MatrixMultiplicationStreamingData.c	0x4124f0
// multipy_sse			1.249s		4,352,400,000	6, 179, 400, 000		0.704		38.5 %		0.0 %     0.0 %					62.8 %			3.5 GHz 				mxm_streaming_data.exe	multipy_sse			MatrixMultiplicationStreamingData.c	0x4127c0
// multipy_avx			0.468s		1,742,400,000	4, 163, 400, 000		0.419		38.0 %		0.8 %     4.6 %					56.6 %			3.7 GHz 				mxm_streaming_data.exe	multipy_avx			MatrixMultiplicationStreamingData.c	0x412200
// random_double		0.041s		181,800,000		250, 200, 000			0.727		74.3 %		14.9 %    0.0 %					55.4 %			4.4 GHz 				mxm_streaming_data.exe	random_double		MatrixMultiplicationStreamingData.c	0x412f80
// initMatrixD			0.009s		36,000,000		28, 800, 000			1.250		75.0 %		75.0 %    100.0 %				0.0 %			3.8 GHz 				mxm_streaming_data.exe	initMatrixD			MatrixMultiplicationStreamingData.c	0x411bd0
// copyMatrixDF			0.007s		27,000,000		52, 200, 000			0.517		0.0 %		0.0 %     0.0 %					100.0 %			3.8 GHz 				mxm_streaming_data.exe	copyMatrixDF		MatrixMultiplicationStreamingData.c	0x411910
// initMatrixF			0.005s		18,000,000		28, 800, 000			0.625		0.0 %		0.0 %     0.0 %					100.0 %			3.3 GHz 				mxm_streaming_data.exe	initMatrixF			MatrixMultiplicationStreamingData.c	0x411d20
// copyMatrixDS			0.005s		14,400,000		36, 000, 000			0.400		0.0 %		0.0 %     0.0 %					100.0 %			3.1 GHz 				mxm_streaming_data.exe	copyMatrixDS		MatrixMultiplicationStreamingData.c	0x411a90
// copyMatrixDI			0.002s		7,200,000		18, 000, 000			0.400		0.0 %		0.0 %     0.0 %					100.0 %			4.6 GHz 				mxm_streaming_data.exe	copyMatrixDI		MatrixMultiplicationStreamingData.c	0x4119d0


// RELEASE
// Function/CallStack	CPU Time	Clockticks		Instructions Retired	CPI Rate	Retiring	Front-End Bound	Bad Speculation	Back-End Bound	Average CPU Frequency	Module				    Function (Full)	Source File							Start Address
// multipy				3.166s		13,559,400,000	8, 616, 600, 000		1.574		13.7 %		2.0 %	  0.4 %					83.9 %			4.3 GHz 				mxm_streaming_data.exe	multipy			MatrixMultiplicationStreamingData.c	0x401050
// multipy_mmx			3.101s		12,585,600,000	6, 741, 000, 000		1.867		15.0 %		0.8 %	  0.0 %					84.4 %			4.1 GHz 				mxm_streaming_data.exe	multipy_mmx		MatrixMultiplicationStreamingData.c	0x401210
// multipy_sse2			1.590s		5,414,400,000	4, 842, 000, 000		1.118		20.4 %		0.7 %     0.5 %					78.3 %			3.4 GHz 				mxm_streaming_data.exe	multipy_sse2	MatrixMultiplicationStreamingData.c	0x401440
// multipy_sse			0.495s		1,733,400,000	2, 421, 000, 000		0.716		20.2 %		0.0 %	  0.0 %					79.8 %			3.5 GHz 				mxm_streaming_data.exe	multipy_sse		MatrixMultiplicationStreamingData.c	0x401370
// multipy_avx			0.151s		552,600,000		2, 188, 800, 000		0.252		78.2 %		3.3 %	  6.5 %					12.1 %			3.7 GHz 				mxm_streaming_data.exe	multipy_avx		MatrixMultiplicationStreamingData.c	0x401510
// singleRun			0.027s		102,600,000		160, 200, 000			0.640		0.0 %		0.0 %	  0.0 %					100.0 %			3.9 GHz 				mxm_streaming_data.exe	singleRun		MatrixMultiplicationStreamingData.c	0x401730
// copyMatrixDF			0.002s		12,600,000		14, 400, 000			0.875		0.0 %		0.0 %	  0.0 %					100.0 %			8.1 GHz 				mxm_streaming_data.exe	copyMatrixDF	MatrixMultiplicationStreamingData.c	0x401680

void main(void)
{
	int N = 1024;

	// Run matrix multiplication from 8 to N
	//runAndSaveMxM(N);

	// Run Matrix multiplication NxN 
	singleRun(N);

	//_getch();
}