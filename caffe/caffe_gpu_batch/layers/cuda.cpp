int gpu_index = 0;


#include "cuda.hpp"
#include <stdlib.h>
#include <time.h>
#include"../netsolve.hpp"


extern void caffe_gpu_set(const int N, const float alpha, float* Y);
extern void caffe_gpu_set_int(const int N, const int alpha, int* Y);


void cuda_set(const int N, const float alpha, float* Y)
{
 	caffe_gpu_set(N, alpha, Y);
}


void cuda_set_int(const int N, const int alpha, int* Y)
{
 	caffe_gpu_set_int(N, alpha, Y);
}

void print_gpu_data(float * cudaData, int datalen)
{
	float* data_host=new float[datalen];
	cuda_pull_array(cudaData, data_host, datalen);
	for(int i=0;i<datalen;i++){
		printf("%f,", data_host[i]);
		}
	delete [] data_host;
	printf("\ndata dim=[%d,%d,%d]\n",1,1,datalen);
	getchar();
}


void print_gpu_data_2d(float * cudaData, int h, int w)
{
	float* data_host=new float[h*w];
	cuda_pull_array(cudaData, data_host, h*w);
	for(int r=0;r<h;r++){
		for(int c=0;c<w;c++){
			printf("%f,", data_host[r*w+c]);
			}
		printf("\n");
		}
	delete [] data_host;
	printf("data dim=[%d,%d,%d]\n",1,h,w);
	getchar();
}

void print_gpu_data_3d(float * cudaData, int ch,int h, int w)
{
	float* data_host=new float[ch*h*w];
	cuda_pull_array(cudaData, data_host, ch*h*w);
	for(int i=0;i<ch;i++){
		for(int r=0;r<h;r++){
			for(int c=0;c<w;c++){
				printf("%f,", data_host[i*h*w+r*w+c]);
				}
			printf("\n");
			}
		printf("\n------\n");
		}
	delete [] data_host;
	printf("data dim=[%d,%d,%d]\n",ch,h,w);
	getchar();
}



float *cuda_make_float_array(size_t n)
{
    float *x_gpu;
    size_t size = sizeof(float)*n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
    return x_gpu;
}

void cuda_set_device(int n)
{
    gpu_index = n;
    cudaError_t status = cudaSetDevice(n);
    check_error(status);
}

int cuda_get_device()
{
    int n = 0;
    cudaError_t status = cudaGetDevice(&n);
    check_error(status);
    return n;
}

void check_error(cudaError_t status)
{
    cudaDeviceSynchronize();
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA1 Error: %s\n", s);
        snprintf(buffer, 256, "CUDA Error: %s", s);
		getchar();
    }
    if (status2 != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA2 Error Prev: %s\n", s);
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
	getchar();

    }
}

dim3 cuda_gridsize(size_t n){
    size_t k = (n-1) / BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
    }
    dim3 d = {x, y, 1};
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
    return d;
}

#ifdef CUDNN
cudnnHandle_t cudnn_handle()
{
    static int init[16] = {0};
    static cudnnHandle_t handle[16];
    int i = cuda_get_device();
    if(!init[i]) {
        cudnnCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
}
#endif

cublasHandle_t blas_handle()
{
    static int init[16] = {0};
    static cublasHandle_t handle[16];
    int i = cuda_get_device();
    if(!init[i]) {
        cublasCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
}

float *cuda_make_array(float *x, size_t n)
{
    float *x_gpu;
    size_t size = sizeof(float)*n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
    if(x){
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        check_error(status);
    }
    if(!x_gpu) printf("Cuda malloc failed\n");
    return x_gpu;
}

//void cuda_random(float *x_gpu, size_t n)
//{
//    static curandGenerator_t gen[16];
//    static int init[16] = {0};
//    int i = cuda_get_device();
//    if(!init[i]){
//        curandCreateGenerator(&gen[i], CURAND_RNG_PSEUDO_DEFAULT);
//        curandSetPseudoRandomGeneratorSeed(gen[i], time(0));
//        init[i] = 1;
//    }
//    curandGenerateUniform(gen[i], x_gpu, n);
//    check_error(cudaPeekAtLastError());
//}

//float cuda_compare(float *x_gpu, float *x, size_t n, char *s)
//{
//    float *tmp = calloc(n, sizeof(float));
//    cuda_pull_array(x_gpu, tmp, n);
//    //int i;
//    //for(i = 0; i < n; ++i) printf("%f %f\n", tmp[i], x[i]);
//    axpy_cpu(n, -1, x, 1, tmp, 1);
//    float err = dot_cpu(n, tmp, 1, tmp, 1);
//    printf("Error %s: %f\n", s, sqrt(err/n));
//    free(tmp);
//    return err;
//}

int *cuda_make_int_array(size_t n)
{
    int *x_gpu;
    size_t size = sizeof(int)*n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
    return x_gpu;
}

void cuda_free(void *x_gpu)
{
    cudaError_t status = cudaFree(x_gpu);
    check_error(status);
}

void cuda_push_array(float *x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float)*n;
    cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
    check_error(status);
}

void cuda_pull_array(float *x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float)*n;
    cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
    check_error(status);
}

//float cuda_mag_array(float *x_gpu, size_t n)
//{
//    float *temp = calloc(n, sizeof(float));
//    cuda_pull_array(x_gpu, temp, n);
//    float m = mag_array(temp, n);
//    free(temp);
//    return m;
//}

