#include<stdio.h>
#include<cuda_runtime.h>
#include"../netsolve.hpp"
#include <cfloat>



#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)


__global__
void im2col_gpu_kernel(const int n, const float* data_im,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col,
        float *data_col) {

    int index = blockIdx.x*blockDim.x+threadIdx.x;
    for(; index < n; index += blockDim.x*gridDim.x){
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;

        float* data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        const float* data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;

                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width + j] : 0;

                data_col_ptr += height_col * width_col;
            }
        }
    }
}



__global__ void col2im_gpu_kernel(const int n, const float* data_col,
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    float* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    float val = 0;
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    const int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = min(w_im / stride_w + 1, width_col);
    const int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = min(h_im / stride_h + 1, height_col);
    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = (h_im - h_col * stride_h);
        int w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                                height_col + h_col) * width_col + w_col;
          val += data_col[data_col_index];
        }
      }
    }
    data_im[index] = val;
  }
}


__global__ void PReLUForward(const int n, const int channels, const int dim,
    const float* in, float* out, const float* slope_data,
    const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[c];
  }
}

__global__ void PReLUParamBackward(const int n, const float* top_diff,
    const float* bottom_data, float* slope_diff, const int div_factor, const int channels, const int dim) {
  CUDA_KERNEL_LOOP(index, n) {
	int c = (index / dim) % channels / div_factor;
	slope_diff[c] += top_diff[index] * bottom_data[index] * (bottom_data[index] <= 0);
  }
}

__global__ void PReLUBackward(const int n, const int channels, const int dim,
    const float* in_diff, const float* in_data, float* out_diff,
    const float* slope_data, const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * slope_data[c]);
  }
}


void PreLULayer_Forward_gpu(const int count, const int channels, const int dim,
							const float* bottom_data, float* top_data, const float* slope_data,
							const int div_factor, int num_kernels)
{
	PReLUForward<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(count, channels, dim, bottom_data, top_data, slope_data, div_factor);
        //cudaDeviceSynchronize();
}

void PreLULayer_Backward_gpu(const int count, const int channels, const int dim,
							const float* bottom_data, float* bottom_diff, float* top_diff, float* slope_diff, const float* slope_data,
							const int div_factor, int num_kernels)
{
	PReLUParamBackward<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(channels*dim,top_diff,bottom_data, slope_diff, div_factor, channels, dim);

	PReLUBackward<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(count, channels, dim, top_diff, bottom_data, bottom_diff, slope_data, div_factor);
        //cudaDeviceSynchronize();
}


__global__ void ReLUForward(const int n, const float* in, float* out,
    float negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}


__global__ void ReLUBackward(const int n, const float* in_diff,
    const float* in_data, float* out_diff, float negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * negative_slope);
  }
}



__global__ void set_kernel(const int n, const float alpha, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}


__global__ void set_int_kernel(const int n, const int alpha, int* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

__global__ void mul_kernel(const int n, const float* a,
    const float* b, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

__global__ void sqrt_kernel(const int n, const float* a, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = sqrt(a[index]);
  }
}


__global__ void div_kernel(const int n, const float* a,
    const float* b, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}


__global__ void add_scalar_kernel(const int n, const float alpha, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

void caffe_gpu_add_scalar(const int num_kernels, const float alpha, float* Y) {
  add_scalar_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>( num_kernels, alpha, Y);
}



void caffe_gpu_div(const int num_kernels, const float* a,
    const float* b, float* y) {
  div_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>( num_kernels, a, b, y);
}

void caffe_gpu_sqrt(const int num_kernels, const float* a, float* y) {
  sqrt_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>( num_kernels, a, y);
}


void caffe_gpu_mul(const int num_kernels, const float* a,
    const float* b, float* y) {
  mul_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>( num_kernels, a, b, y);
}




void caffe_gpu_set(const int num_kernels, const float alpha, float* Y) {
  if (alpha == 0) {
    cudaMemset(Y, 0, sizeof(float) * num_kernels);  // NOLINT(caffe/alt_fn)
    return;
  }
  set_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>( num_kernels, alpha, Y);

        //cudaDeviceSynchronize();
}


void caffe_gpu_set_int(const int num_kernels, const int alpha, int* Y) {
  if (alpha == 0) {
    cudaMemset(Y, 0, sizeof(int) * num_kernels);  // NOLINT(caffe/alt_fn)
    return;
  }
  set_int_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>( num_kernels, alpha, Y);

        //cudaDeviceSynchronize();
}




void ReLULayer_Forward_gpu(float negative_slope, float* bottom_data, float* top_data, int num_kernels)
{
  ReLUForward<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>( num_kernels, bottom_data, top_data, negative_slope);

        //cudaDeviceSynchronize();
}

void ReLULayer_Backward_gpu(float negative_slope, float* top_diff, float* bottom_data, float* bottom_diff,int num_kernels)
{
    ReLUBackward<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(num_kernels, top_diff, bottom_data, bottom_diff, negative_slope);

        //cudaDeviceSynchronize();
}


void im2col_ongpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad, float *data_col){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int count = channels * height_col * width_col;

    im2col_gpu_kernel<<<(count+BLOCK-1)/BLOCK,
        BLOCK>>>(
                count, im, height, width, ksize, pad,
                stride, height_col,
                width_col, data_col);
    //cudaDeviceSynchronize();
}

void col2im_gpu(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_im) {
  int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) /
      stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) /
      stride_w + 1;
  int num_kernels = channels * height * width;
  col2im_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(num_kernels, data_col, height, width, channels, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      height_col, width_col, data_im);

      //cudaDeviceSynchronize();

}











__global__ void MaxPoolForward(const int nthreads,
    const float* const bottom_data, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    float* const top_data, int* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    float maxval = -FLT_MAX;
    int maxidx = -1;
    const float* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_slice[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    mask[index] = maxidx;
  }
}


void PoolingLayer_Forward_gpu(t_layer_info*  layer_list,int layer_id, float* input, int ch, int h, int w,float* output, int *mask)
{

	int channels_=ch;
	int height_=h;
	int width_=w;
	int pooled_height_=layer_list[layer_id].out_dim[1];
	int pooled_width_=layer_list[layer_id].out_dim[2];
	int stride_h_=layer_list[layer_id].ker_nfo.stride;
	int stride_w_=layer_list[layer_id].ker_nfo.stride;
	int pad_h_=layer_list[layer_id].ker_nfo.pad;
	int pad_w_=layer_list[layer_id].ker_nfo.pad;
	int kernel_h_=layer_list[layer_id].ker_nfo.ker_size;
	int kernel_w_=layer_list[layer_id].ker_nfo.ker_size;
	int num_kernels = pooled_height_*pooled_width_*channels_;

  switch (layer_list[layer_id].ker_nfo.act) {
	  case pool_max:
		    MaxPoolForward<<<(num_kernels+BLOCK-1)/BLOCK, BLOCK>>>(
		        num_kernels, input, channels_,
		        height_, width_, pooled_height_, pooled_width_, kernel_h_,
		        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, output,
		        mask);
		    break;
	  default:
		    printf("Unknown pooling method.\n");
		    break;
	  }

	  //cudaDeviceSynchronize();
}









__global__ void MaxPoolBackward(const int nthreads, const float* const top_diff,
    const int* const mask,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, float* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart =
         (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
         (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    float gradient = 0;
    const int offset = (n * channels + c) * pooled_height * pooled_width;
    const float* const top_diff_slice = top_diff + offset;
      const int* const mask_slice = mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }

    bottom_diff[index] = gradient;
  }
}





void PoolingLayer_Backward_gpu(t_layer_info*  layer_list,int layer_id, int *mask,float* bottom, float* top)
{

	int bottom_size=layer_list[layer_id].in_dim[0][2]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][0];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	float* bottom_diff=bottom+bottom_size;
	float* top_diff=top+top_size;

	int channels_=layer_list[layer_id].in_dim[0][0];
	int height_=layer_list[layer_id].in_dim[0][1];
	int width_=layer_list[layer_id].in_dim[0][2];
	int pooled_height_=layer_list[layer_id].out_dim[1];
	int pooled_width_=layer_list[layer_id].out_dim[2];
	int stride_h_=layer_list[layer_id].ker_nfo.stride;
	int stride_w_=layer_list[layer_id].ker_nfo.stride;
	int pad_h_=layer_list[layer_id].ker_nfo.pad;
	int pad_w_=layer_list[layer_id].ker_nfo.pad;
	int kernel_h_=layer_list[layer_id].ker_nfo.ker_size;
	int kernel_w_=layer_list[layer_id].ker_nfo.ker_size;
	int num_kernels = bottom_size;


	switch (layer_list[layer_id].ker_nfo.act) {
	  case pool_max:
		   MaxPoolBackward<<<(num_kernels+BLOCK-1)/BLOCK, BLOCK>>>(
		        num_kernels, top_diff, mask,   channels_,
		        height_, width_, pooled_height_, pooled_width_,
		        kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
		        bottom_diff);
		    break;
	  default:
		    printf("Unknown pooling method.\n");
		    break;
	  }

	//cudaDeviceSynchronize();
}







#if 0
__global__ void add_bias_kernel(float *output, float *biases, int n, int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;

    if(offset < size) output[(batch*n+filter)*size + offset] += biases[filter];
}

void add_bias_gpu(float *output, float *biases, int ch, int dim_size)
{
    dim3 dimGrid((dim_size-1)/BLOCK + 1, ch, batch);
    dim3 dimBlock(BLOCK, 1, 1);

    add_bias_kernel<<<dimGrid, dimBlock>>>(output, biases, ch, dim_size);
    check_error(cudaPeekAtLastError());
}
#endif


__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x*blockDim.x+threadIdx.x;
  int stride = blockDim.x*gridDim.x;
  for (int i = index; i < n; i += stride){
      y[i] = x[i] + y[i];
      }

}



int cuda_add_test(int N, float* x, float* y)
{

  // Run kernel on 1M elements on the GPU
  add<<<1, 256>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  std::cout << "Max error: " << maxError << std::endl;

  return 0;
}


float* transfer_data2cuda(float* data,int data_size)
{
	float* x;
	cudaMallocManaged(&x, data_size*sizeof(float));
	cudaMemcpy(x,data,data_size*sizeof(float),cudaMemcpyHostToDevice);
	return x;
}

float* malloc_buffer_in_cuda(int data_size)
{
	float* x;
	cudaMallocManaged(&x, data_size*sizeof(float));
	return x;
}

