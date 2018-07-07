#ifndef _CUDA_H
#define _CUDA_H

extern int gpu_index;

#include <stdio.h>
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"


void check_error(cudaError_t status);
cublasHandle_t blas_handle();
float *cuda_make_array(float *x, size_t n);
int *cuda_make_int_array(size_t n);
void cuda_push_array(float *x_gpu, float *x, size_t n);
void cuda_pull_array(float *x_gpu, float *x, size_t n);
void cuda_set_device(int n);
void cuda_free(void *x_gpu);
//void cuda_random(float *x_gpu, size_t n);
//float cuda_compare(float *x_gpu, float *x, size_t n, char *s);
dim3 cuda_gridsize(size_t n);
//float cuda_mag_array(float *x_gpu, size_t n);
void cuda_set(const int N, const float alpha, float* Y);
float *cuda_make_float_array(size_t n);
extern 	void caffe_gpu_mul(const int N, const float* a,  const float* b, float* y);
extern	void caffe_gpu_sqrt(const int N, const float* a, float* y);
extern	void caffe_gpu_div(const int N, const float* a,  const float* b, float* y);
extern	void cuda_set_int(const int N, const int alpha, int* Y);
extern 	void print_gpu_data(float * cudaData, int datalen);
extern 	void caffe_gpu_set(const int num_kernels, const float alpha, float* Y);
extern	void print_gpu_data_2d(float * cudaData, int h, int w);
extern	void print_gpu_data_3d(float * cudaData, int ch,int h, int w);
extern	void caffe_gpu_add_scalar(const int N, const float alpha, float* Y);
#endif
