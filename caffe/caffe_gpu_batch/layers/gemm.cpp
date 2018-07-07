#include"gemm.hpp"
#include<string.h>
#include <math.h>
//#include"../netsolve.hpp"
#include"cuda.hpp"

cublasHandle_t cublasHandle;

void caffe_ini_cublas(void)
{
	cublasCreate(&cublasHandle);
}

void caffe_cpu_scale(const int n, const float alpha, const float *x,
                            float* y) {
  	cblas_scopy(n, x, 1, y, 1);
  	cblas_sscal(n, alpha, y, 1);
}


void caffe_scal(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}


void caffe_gpu_scal(const int N, const float alpha, float *X) {
  cublasSscal(cublasHandle, N, &alpha, X, 1);
}



void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasSgemm(cublasHandle, cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N);
}



void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
	  int lda = (TransA == CblasNoTrans) ? K : M;
	  int ldb = (TransB == CblasNoTrans) ? N : K;
	  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
	      ldb, beta, C, N);
}


void caffe_set(const int N, const float alpha, float* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(float) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}



void caffe_gpu_copy(const int N, const float* X, float* Y) {
  if (X != Y) {
   cudaMemcpy(Y, X, N*sizeof(float), cudaMemcpyDefault);  // NOLINT(caffe/alt_fn)
  }
}



void caffe_copy(const int N, const float* X, float* Y) {
  if (X != Y) {
      memcpy(Y, X, sizeof(float) * N);  // NOLINT(caffe/alt_fn)
  }
}

float caffe_cpu_strided_dot(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}


float caffe_cpu_dot(const int n, const float* x, const float* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}



void caffe_gpu_axpy(const int N, const float alpha, const float* X,
    float* Y) {
  cublasSaxpy(cublasHandle, N, &alpha, X, 1, Y, 1);
}


void caffe_axpy(const int N, const float alpha, const float* X, float* Y) {
	cblas_saxpy(N, alpha, X, 1, Y, 1);
}


static void cblas_saxpby(const int N, const float alpha, const float* X,
                         const int incX, const float beta, float* Y,
                         const int incY) {
  cblas_sscal(N, beta, Y, incY);
  cblas_saxpy(N, alpha, X, incX, Y, incY);
}

void caffe_cpu_axpby(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}


void caffe_gpu_axpby(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal(N, beta, Y);
  caffe_gpu_axpy(N, alpha, X, Y);
}

void caffe_mul(const int n, const float* a, const float* b, float* y)
{
  for(int i=0;i<n;i++){
	y[i]=a[i]*b[i];
  	}
}
