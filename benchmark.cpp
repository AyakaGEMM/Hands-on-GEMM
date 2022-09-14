#include <stdlib.h> // For: exit, drand48, malloc, free, NULL, EXIT_FAILURE
#include <stdio.h>  // For: perror
#include <string.h> // For: memset
#include <iostream>

#include <float.h> // For: DBL_EPSILON
#include <math.h>  // For: fabs

#ifdef GETTIMEOFDAY
#include <sys/time.h> // For struct timeval, gettimeofday
#else
#include <time.h> // For struct timespec, clock_gettime, CLOCK_MONOTONIC
#endif

/* reference_gemm wraps a call to the BLAS-3 routine GEMM, via the standard FORTRAN interface - hence the reference semantics. */
void reference_gemm(int N, int K, float ALPHA, float *A, float *B, float *C)
{
    int M = 6;
    int LDA = K;
    int LDB = N;
    int LDC = N;
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < K; k++)
                C[i * LDC + j] += ALPHA * A[i * LDA + k] * B[k * LDB + j];
}

/* Your function must have the following signature: */
extern const char *gemm_desc;
extern void square_gemm(int, int, int, float *, float *, float *);
extern void gemm_compute(int, int, int, float *, float *, float *, bool beta = false);
extern void gemm_layernorm_compute_sum(int, int, int, float *, float *, float *, float *, float *, const bool &beta = false, float eps = 1e-5);
extern float *packing(int, int, float *, int);
extern void free_packing(float *);

double wall_time()
{
#ifdef GETTIMEOFDAY
    struct timeval t;
    gettimeofday(&t, NULL);
    return 1. * t.tv_sec + 1.e-6 * t.tv_usec;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return 1. * t.tv_sec + 1.e-9 * t.tv_nsec;
#endif
}

void die(const char *message)
{
    perror(message);
    exit(EXIT_FAILURE);
}

void fill(float *p, int n)
{
    for (int i = 0; i < n; ++i)
        p[i] = 2 * drand48() - 1; // Uniformly distributed over [-1, 1]
}

void absolute_value(float *p, int n)
{
    for (int i = 0; i < n; ++i)
        p[i] = fabs(p[i]);
}

/* The benchmarking program */
int main(int argc, char **argv)
{
    printf("Description:\t%s\n\n", gemm_desc);

    /* Test sizes should highlight performance dips at multiples of certain powers-of-two */

    int test_sizes[] =

        /* Multiples-of-32, +/- 1. for final benchmarking. */
        //  {31,32,33,63,64,65,95,96,97,127,128,129,159,160,161,191,192,193,223,224,225,255,256,257,287,288,289,319,320,321,351,352,353,383,384,385,415,416,417,447,448,449,479,480,481,511,512,513,543,544,545,575,576,577,607,608,609,639,640,641,671,672,673,703,704,705,735,736,737,767,768,769,799,800,801,831,832,833,863,864,865,895,896,897,927,928,929,959,960,961,991,992,993,1023,1024,1025};

        /* A representative subset of the first list for initial test. Currently uncommented. */
        //  { 31, 32, 96, 97, 127, 128, 129, 191, 192, 229, 255, 256, 257,
        //    319, 320, 321, 417, 479, 480, 511, 512, 639, 640, 767, 768, 769 };
        // {16};
        // {31, 32};

        //  { 31, 32, 96, 97, 127, 128, 129, 191, 192, 229, 255, 256, 257,
        //    319, 320, 321, 417, 479, 480, 511, 512, 639, 640, 767, 768, 769, 1024, 2048 };

        {127, 128, 129, 255, 256, 257, 383, 384, 385, 511, 512, 513, 639, 640,
         641, 767, 768, 769, 895, 896, 897, 1023, 1024, 1025, 1151, 1152, 1153, 1279, 1280, 1281, 2047, 2048, 2049, 4095, 4096, 4097, 1024, 8192};

    int nsizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    /* assume last size is also the largest size */
    int nmax = test_sizes[nsizes - 1];

    /* allocate memory for all problems */
    float *buf = NULL;
    buf = (float *)malloc(3 * nmax * nmax * sizeof(float));
    if (buf == NULL)
        die("failed to allocate largest problem size");

    /* For each test size */
    double res = 0.0;
    double count = 0.0;
    for (int isize = 0; isize < sizeof(test_sizes) / sizeof(test_sizes[0]); ++isize)
        for (int jsize = 0; jsize < sizeof(test_sizes) / sizeof(test_sizes[0]); ++jsize)
        {
            /* Create and fill 3 random matrices A,B,C*/
            int n = test_sizes[isize];
            int k = test_sizes[jsize];

            float *A = buf + 0;
            float *B = A + nmax * nmax;
            float *C = B + nmax * nmax;

            fill(A, 6 * k);
            fill(B, k * n);
            fill(C, 6 * n);

            auto newB = packing(n, k, B, n);

            /* Measure performance (in Gflops/s). */

            /* Time a "sufficiently long" sequence of calls to reduce noise */
            double Gflops_s, seconds = -1.0, fused_seconds = -1.0, Fused_gflops_s = -1.0;
            double timeout = 0.1; // "sufficiently long" := at least 1/10 second.
            int n_iterations = 0;

            for (n_iterations = 1; seconds < timeout / 2; n_iterations *= 2)
            {
                /* Warm-up */
                gemm_compute(6, n, k, A, newB, C);
                gemm_layernorm_compute_sum(6, n, k, A, newB, C, A, A);

                /* Benchmark n_iterations runs of square_gemm */
                seconds = -wall_time();
                for (int it = 0; it < n_iterations; ++it)
                    gemm_compute(6, n, k, A, newB, C);
                seconds += wall_time();

                fused_seconds = -wall_time();
                for (int it = 0; it < n_iterations; ++it)
                    gemm_layernorm_compute_sum(6, n, k, A, newB, C, A, A);
                fused_seconds += wall_time();

                /*  compute Mflop/s rate */
                Gflops_s = 2.e-9 * n_iterations * 6 * k * n / seconds;
                Fused_gflops_s = 2.e-9 * n_iterations * 6 * k * n / fused_seconds;
            }
            // gptlRet = GPTLstop("gemm");
            // gptlRet = GPTLpr_file("outfile");
            printf("Size (n, k): (%d, %d)\tGflop/s: %.3g (%d iter, %.3f seconds)\t\tFused Gflop/s: %.3g (%d iter, %.3f seconds)\n", n, k, Gflops_s, n_iterations, seconds, Fused_gflops_s, n_iterations, fused_seconds);
            res += Gflops_s;
            count += 1;
            /* Ensure that error does not exceed the theoretical error bound. */

            /* C := A * B, computed with square_gemm */
            memset(C, 0, 6 * n * sizeof(float));
            gemm_compute(6, n, k, A, newB, C);
            free_packing(newB);

            /* Do not explicitly check that A and B were unmodified on square_gemm exit
             *  - if they were, the following will most likely detect it:
             * C := C - A * B, computed with reference_gemm */
            reference_gemm(n, k, -1., A, B, C);

            /* A := |A|, B := |B|, C := |C| */
            absolute_value(A, 6 * k);
            absolute_value(B, k * n);
            absolute_value(C, 6 * n);

            /* C := |C| - 3 * e_mach * n * |A| * |B|, computed with reference_gemm */
            reference_gemm(n, k, -3. * FLT_EPSILON * n, A, B, C);

            /* If any element in C is positive, then something went wrong in square_gemm */
            for (int i = 0; i < n * n; ++i)
                if (C[i] > 0)
                    die("*** FAILURE *** Error in matrix multiply exceeds componentwise error bounds.\n");
        }
    res /= count;
    printf("Average %lf \n", res);

    free(buf);

    return 0;
}