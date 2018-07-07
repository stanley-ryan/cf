#ifndef	_GEMM_H
#define 	_GEMM_H

#include<stdio.h>
#include <cublas_v2.h>
#include<math.h>


extern "C" {
#include <cblas.h>
}

extern	void caffe_ini_cublas(void);

extern	void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C);


extern	void caffe_gpu_axpy(const int N, const float alpha, const float* X,
    float* Y);

extern	void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C);

extern	void caffe_gpu_scal(const int N, const float alpha, float *X);
extern	void caffe_cpu_scale(const int n, const float alpha, const float *x,  float* y);

extern	void caffe_scal(const int N, const float alpha, float *X);

extern	void caffe_set(const int N, const float alpha, float* Y);

extern	void caffe_copy(const int N, const float* X, float* Y);

extern	void caffe_axpy(const int N, const float alpha, const float* X, float* Y);

extern	void caffe_cpu_axpby(const int N, const float alpha, const float* X,    const float beta, float* Y);

extern	void caffe_gpu_copy(const int N, const float* X, float* Y);

extern	void caffe_gpu_axpby(const int N, const float alpha, const float* X,  const float beta, float* Y);


extern void caffe_mul(const int n, const float* a, const float* b, float* y);

#endif
