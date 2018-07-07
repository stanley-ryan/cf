
#include <iostream>
#include <string.h>

#include <stdio.h>
#include <getopt.h>
#include <sys/time.h>
#include <time.h>
#include<pthread.h>

#include "gemm.hpp"
#include "Convxor_layer.hpp"
#include"cuda.hpp"

#if 1
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}
#else
int is_a_ge_zero_and_a_lt_b(int a, int b) {
  return (unsigned)(a) < (unsigned)(b);
}
#endif





typedef struct ker_product_thread_para{
	t_layer_info*  layer_list;
	int layer_id;
	float* input;
	int h;
	int w;
	float* output;
	 int start_r;
	 int end_r;
}t_ker_product_thread_para;






static void kernel_product_normal(t_layer_info*  layer_list,int layer_id,float* input, int h, int w, float* output)
{
	int r,c;
	int i;
	int ker_matrix_h  = layer_list[layer_id].output_num;
	int ker_matrix_w  = h;

	caffe_gpu_gemm(CblasNoTrans, CblasNoTrans,ker_matrix_h,w, h, 1, layer_list[layer_id].para1_entry,input, 0, output);
	if(layer_list[layer_id].train_inf.en_bias){
		float* bias_multiplier_=cuda_make_float_array(w);
		cuda_set(w, 1.0, bias_multiplier_);
		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, ker_matrix_h, w, 1, 1., layer_list[layer_id].para2_entry, bias_multiplier_,	1., output);
		cuda_free(bias_multiplier_);
		}
}


int 	net_convxnor_initial_para(t_layer_info*  layer_list,int layer_id,int in_ch)
{
	int para_size;
	int conv_w_size=layer_list[layer_id].output_num*in_ch*layer_list[layer_id].ker_nfo.ker_size*layer_list[layer_id].ker_nfo.ker_size;
	int fan_in=in_ch*layer_list[layer_id].ker_nfo.ker_size*layer_list[layer_id].ker_nfo.ker_size;
	int fan_out=layer_list[layer_id].output_num*layer_list[layer_id].ker_nfo.ker_size*layer_list[layer_id].ker_nfo.ker_size;
	int o_c=layer_list[layer_id].output_num;


	para_size=conv_w_size;
	float* conv_w=new float[para_size];

	if(layer_list[layer_id].train_inf.para1_ini_way==para_ini_gaussian){
		gaussian_fill(0.01,0.0,conv_w, para_size);
		layer_list[layer_id].para1_entry=cuda_make_array(conv_w,para_size);
		printf("initial para1 conve_layer_%d: initial parameter gaussian\n",layer_id);
	}else if(layer_list[layer_id].train_inf.para1_ini_way==para_ini_xavier){
		float scale=sqrt(3.0/fan_in);
		unirom_fill(-scale, scale, conv_w, para_size);
		layer_list[layer_id].para1_entry=cuda_make_array(conv_w,para_size);
		printf("initial para1 conve_layer_%d: initial parameter uninorm\n",layer_id);
	}else if(layer_list[layer_id].train_inf.para1_ini_way==para_ini_loading){
		layer_list[layer_id].para1_entry=cuda_make_array(layer_list[layer_id].para1_entry,para_size);
		printf("initial para1 conve_layer_%d: loading parameter\n",layer_id);
	}else{
		printf("initial para1 conve_layer_%d: invalid parameter initial way\n",layer_id);
		exit(-1);
		}
	delete [] conv_w;
	gpu_xnor_nomalize_by_rows(layer_list[layer_id].para1_entry, o_c, fan_in);

	if(layer_list[layer_id].train_inf.en_bias){
		para_size=layer_list[layer_id].output_num;
		fan_in=in_ch;
		float* conv_b =new float[para_size];
		if(layer_list[layer_id].train_inf.para2_ini_way==para_ini_gaussian){
			gaussian_fill(0.01,0.0,conv_b, para_size);
			layer_list[layer_id].para2_entry=cuda_make_array(conv_b,para_size);
			printf("initial para2 conve_layer_%d: initial parameter gaussian\n",layer_id);
		}else if(layer_list[layer_id].train_inf.para2_ini_way==para_ini_xavier){
			float scale=sqrt(3.0/fan_in);
			unirom_fill(-scale, scale, conv_b, para_size);
			layer_list[layer_id].para2_entry=cuda_make_array(conv_b,para_size);
			printf("initial para2 conve_layer_%d: initial parameter uninorm\n",layer_id);
		}else if(layer_list[layer_id].train_inf.para2_ini_way==para_ini_loading){
			layer_list[layer_id].para2_entry=cuda_make_array(layer_list[layer_id].para2_entry,para_size);
			printf("initial para2 conve_layer_%d: loading parameter\n",layer_id);
		}else if(layer_list[layer_id].train_inf.para2_ini_way==para_ini_const_zero){
			layer_list[layer_id].para2_entry=cuda_make_float_array(para_size);
			cuda_set(para_size, 0.0, layer_list[layer_id].para2_entry);
		}else{
			printf("initial para2 conve_layer_%d: invalid parameter initial way\n",layer_id);
			exit(-1);
			}
		delete [] conv_b;
		}

 	if(layer_list[layer_id].train_inf.para_diff_way==e_para_unchange){
		//layer_list[layer_id].train_inf.para_diff_mask=new float[conv_w_size];
		//caffe_set(conv_w_size,0,layer_list[layer_id].train_inf.para_diff_mask);
		printf("layer id =%d, all zero for para1/2_diff\n",layer_id);
		}

	layer_list[layer_id].train_inf.para1_diff=cuda_make_float_array(conv_w_size);
	layer_list[layer_id].train_inf.para1_diff_hist=cuda_make_float_array(conv_w_size);
	layer_list[layer_id].train_inf.para1_size=conv_w_size;
	if(layer_list[layer_id].train_inf.en_bias){
		layer_list[layer_id].train_inf.para2_diff=cuda_make_float_array(layer_list[layer_id].output_num);
		layer_list[layer_id].train_inf.para2_diff_hist=cuda_make_float_array(layer_list[layer_id].output_num);
		layer_list[layer_id].train_inf.para2_size=layer_list[layer_id].output_num;
		}

	cuda_set(conv_w_size,0, layer_list[layer_id].train_inf.para1_diff);
	cuda_set(conv_w_size,0, layer_list[layer_id].train_inf.para1_diff_hist);
	if(layer_list[layer_id].train_inf.en_bias){
		cuda_set(layer_list[layer_id].output_num,0, layer_list[layer_id].train_inf.para2_diff);
		cuda_set(layer_list[layer_id].output_num,0, layer_list[layer_id].train_inf.para2_diff_hist);
		}

	return 0;
}


extern	void im2col_ongpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad, float *data_col);


int 	net_convxnor_forward(t_layer_info*  layer_list,int layer_id, float* bottom, int ch, int h, int w,float* top)
{
	//MEASURE_TIME_VARIABLES();
	int o_h;
	int o_w;
	int o_c;
	int im2col_buf_size=0;
	float* im2col_buf;

	o_c=layer_list[layer_id].out_dim[0];
	o_h=layer_list[layer_id].out_dim[1];
	o_w=layer_list[layer_id].out_dim[2];

	int im2col_w=o_h*o_w;
	int im2col_h=ch*layer_list[layer_id].ker_nfo.ker_size*layer_list[layer_id].ker_nfo.ker_size;
	int keyprod_h=layer_list[layer_id].out_dim[0];

	im2col_buf_size=im2col_h * im2col_w;

	im2col_buf=cuda_make_float_array(im2col_buf_size);
	im2col_ongpu(bottom,ch,h,w,layer_list[layer_id].ker_nfo.ker_size,layer_list[layer_id].ker_nfo.stride,layer_list[layer_id].ker_nfo.pad,(float*)im2col_buf);

	gpu_xnor_nomalize_by_cols(im2col_buf,im2col_h,im2col_w);

	kernel_product_normal(layer_list, layer_id,im2col_buf,im2col_h, im2col_w, top);

	cuda_free(im2col_buf);

	return 1;
}


int net_convxnor_forward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz)
{
	int i_c=layer_list[layer_id].in_dim[0][0];
	int i_h=layer_list[layer_id].in_dim[0][1];
	int i_w=layer_list[layer_id].in_dim[0][2];

	int bottom_size=layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	for(int i=0;i<batch_sz;i++){
		float* bottom_=layer_list[layer_id].input_buf[0]+i*3*bottom_size;
		float* top_=layer_list[layer_id].output_buf+i*3*top_size;
		net_convxnor_forward( layer_list, layer_id, bottom_, i_c, i_h, i_w,  top_);
		}
}



extern	void col2im_gpu(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_im);


int 	net_convxnor_backward(t_layer_info*  layer_list,int layer_id, float* bottom, float* top)
{
	int o_c=layer_list[layer_id].out_dim[0];
	int o_h=layer_list[layer_id].out_dim[1];
	int o_w=layer_list[layer_id].out_dim[2];

	int i_c=layer_list[layer_id].in_dim[0][0];
	int i_h=layer_list[layer_id].in_dim[0][1];
	int i_w=layer_list[layer_id].in_dim[0][2];
	int bottom_size_h=i_c;
	int bottom_size_w=i_h*i_w;
	int bottom_size=bottom_size_h*bottom_size_w;
	float* bottom_diff=bottom+bottom_size;

	int top_size_h=o_c;
	int top_size_w=o_h*o_w;
	float* top_diff=top+top_size_h*top_size_w;

	int im2col_w=o_h*o_w;
	int im2col_h=i_c*layer_list[layer_id].ker_nfo.ker_size*layer_list[layer_id].ker_nfo.ker_size;

	int bottom_im2col_size_h=im2col_h;
	int bottom_im2col_size_w=im2col_w;
	int bottom_im2col_size=bottom_im2col_size_h*bottom_im2col_size_w;
	float*	im2col_bottom_data=cuda_make_float_array(im2col_h*im2col_w);


	float* bottom_diff_temp=bottom+bottom_size*2;
	int conv_w_size=layer_list[layer_id].output_num*i_c*layer_list[layer_id].ker_nfo.ker_size*layer_list[layer_id].ker_nfo.ker_size;

	im2col_ongpu(bottom,i_c,
				i_h,	i_w,
				layer_list[layer_id].ker_nfo.ker_size,
				layer_list[layer_id].ker_nfo.stride,
				layer_list[layer_id].ker_nfo.pad,
				im2col_bottom_data);
	caffe_gpu_gemm(CblasNoTrans,
		    CblasTrans, top_size_h,bottom_im2col_size_h,top_size_w,
		    1.0, top_diff, im2col_bottom_data, 1,
		    layer_list[layer_id].train_inf.para1_diff);

	if(layer_list[layer_id].train_inf.en_bias){
		float* bias_multiplier_=cuda_make_float_array(top_size_w);
		cuda_set(top_size_w, 1.0, bias_multiplier_);
		caffe_gpu_gemm(CblasNoTrans,
			    CblasTrans, top_size_h,1,top_size_w,
			    1.0, top_diff, bias_multiplier_, 1.0,
			    layer_list[layer_id].train_inf.para2_diff);
		cuda_free(bias_multiplier_);
		}

	caffe_gpu_gemm(CblasTrans,
		    CblasNoTrans, bottom_im2col_size_h,top_size_w,top_size_h,
		    1, layer_list[layer_id].para1_entry, top_diff, 0,
		    im2col_bottom_data);

	col2im_gpu(im2col_bottom_data,i_c,
				i_h,	i_w,	layer_list[layer_id].ker_nfo.ker_size,	layer_list[layer_id].ker_nfo.ker_size,
				layer_list[layer_id].ker_nfo.pad,	layer_list[layer_id].ker_nfo.pad,
				layer_list[layer_id].ker_nfo.stride,	layer_list[layer_id].ker_nfo.stride,
				layer_list[layer_id].ker_nfo.dialation,	layer_list[layer_id].ker_nfo.dialation,
				bottom_diff_temp);

	caffe_gpu_axpy(bottom_size, 1.0, bottom_diff_temp,bottom_diff);
	cuda_free(im2col_bottom_data);

	if(layer_list[layer_id].train_inf.para_diff_way==e_para_unchange){
		cuda_set(conv_w_size,0,layer_list[layer_id].train_inf.para1_diff);
		if(layer_list[layer_id].train_inf.en_bias){
			cuda_set(layer_list[layer_id].output_num,0,layer_list[layer_id].train_inf.para2_diff);
			}
		}
	//print_gpu_data(bottom_diff, bottom_size);
	//print_stop_sign("conv back");


	return 1;
}


int net_convxnor_backward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz)
{
	int bottom_size=layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	for(int i=0;i<batch_sz;i++){
		float* bottom_=layer_list[layer_id].input_buf[0]+i*3*bottom_size;
		float* top_=layer_list[layer_id].output_buf+i*3*top_size;
		net_convxnor_backward( layer_list,layer_id, bottom_, top_);
		}
}

