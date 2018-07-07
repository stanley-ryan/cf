#include <iostream>
#include <string.h>

#include <stdio.h>
#include <getopt.h>
#include <sys/time.h>
#include <time.h>
#include<pthread.h>
#include "gemm.hpp"
#include "BN_layer.hpp"
#include"cuda.hpp"


int 	net_bn_initial_para(t_layer_info*  layer_list,int layer_id,int in_ch)
{
	int para_size;
	int o_c=layer_list[layer_id].output_num;
	para_size=o_c;
#if (TRAIN_MODE>0)
	layer_list[layer_id].para1_entry=cuda_make_float_array(para_size);
	layer_list[layer_id].para2_entry=cuda_make_float_array(para_size);
#else
	layer_list[layer_id].para1_entry=cuda_make_array(layer_list[layer_id].para1_entry,para_size);
	layer_list[layer_id].para2_entry=cuda_make_array(layer_list[layer_id].para2_entry,para_size);
#endif
	layer_list[layer_id].train_inf.para1_size=para_size;
	layer_list[layer_id].train_inf.para2_size=para_size;
	layer_list[layer_id].train_inf.para1_bp_mode=e_para_unbp;
	layer_list[layer_id].train_inf.para2_bp_mode=e_para_unbp;
	return 0;
}




int 	net_bn_forward(t_layer_info*  layer_list,int layer_id, float* input, int ch, int h, int w,float* output)
{
	int o_h;
	int o_w;
	int o_c;
	float* bottom_data=input;
	float* top_data=output;

	o_c=layer_list[layer_id].out_dim[0];
	o_h=layer_list[layer_id].out_dim[1];
	o_w=layer_list[layer_id].out_dim[2];


#if (TRAIN_MODE==0)
	float* mean_multiplier_=cuda_make_float_array(o_h*o_w);
	cuda_set(o_h*o_w, 1.0, mean_multiplier_);
	float* var_cubic=cuda_make_float_array(o_c*o_h*o_w);

	caffe_gpu_copy(o_c*o_h*o_w, bottom_data, top_data);
	caffe_gpu_gemm(CblasNoTrans,				//step2: broadcast mean and cal(x-mean) dim=ch*h*w
			CblasNoTrans, o_c,o_h*o_w,1,
			-1.0, layer_list[layer_id].para1_entry, mean_multiplier_, 1.0,
			top_data);

	caffe_gpu_gemm(CblasNoTrans,				//step4: broadcast var dim=ch*h*w
		    CblasNoTrans, o_c,o_h*o_w,1,
		    1.0, layer_list[layer_id].para2_entry, mean_multiplier_, 0.0,
		    var_cubic);
	caffe_gpu_div(o_c*o_h*o_w,top_data,var_cubic,top_data);

	cuda_free(mean_multiplier_);
	cuda_free(var_cubic);

#else
	float   mean_factor=(float)1.0/(o_h*o_w);
	float* mean_multiplier_=cuda_make_float_array(o_h*o_w);
	cuda_set(o_h*o_w, 1.0, mean_multiplier_);
	float* var_cubic=cuda_make_float_array(o_c*o_h*o_w);

	caffe_gpu_copy(o_c*o_h*o_w, bottom_data, top_data);

	caffe_gpu_gemm(CblasNoTrans,				//step1: cal mean dim=ch*1
		    CblasNoTrans, o_c,1,o_h*o_w,
		    mean_factor, bottom_data, mean_multiplier_, 0.0,
		    layer_list[layer_id].para1_entry);

	//print_gpu_data(layer_list[layer_id].para1_entry, o_c);

	caffe_gpu_gemm(CblasNoTrans,				//step2: broadcast mean and cal(x-mean) dim=ch*h*w
		    CblasNoTrans, o_c,o_h*o_w,1,
		    -1.0, layer_list[layer_id].para1_entry, mean_multiplier_, 1.0,
		    top_data);

	caffe_gpu_mul(o_c*o_h*o_w,top_data,top_data,var_cubic);		//step3: cal (X-Mean)^2		dim=ch*h*w
	caffe_gpu_gemm(CblasNoTrans,							//step4: cal var Mean((X-Mean)^2)	dim=ch*1
		    CblasNoTrans, o_c,1,o_h*o_w,
		    mean_factor, var_cubic, mean_multiplier_, 0.0,
		    layer_list[layer_id].para2_entry);
	caffe_gpu_add_scalar(o_c, EPS_, layer_list[layer_id].para2_entry);		//add eps on var^2+eps_
	caffe_gpu_sqrt(o_c,layer_list[layer_id].para2_entry,layer_list[layer_id].para2_entry);
	caffe_gpu_gemm(CblasNoTrans,				//step4: broadcast var dim=ch*h*w
		    CblasNoTrans, o_c,o_h*o_w,1,
		    1.0, layer_list[layer_id].para2_entry, mean_multiplier_, 0.0,
		    var_cubic);
	caffe_gpu_div(o_c*o_h*o_w,top_data,var_cubic,top_data);
	cuda_free(mean_multiplier_);
	cuda_free(var_cubic);
#endif

	//print_gpu_data(output, o_c*o_h*o_w);

	return 1;
}



int net_bn_forward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz)
{
	int i_c=layer_list[layer_id].in_dim[0][0];
	int i_h=layer_list[layer_id].in_dim[0][1];
	int i_w=layer_list[layer_id].in_dim[0][2];

	int bottom_size=layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	for(int i=0;i<batch_sz;i++){
		float* bottom_=layer_list[layer_id].input_buf[0]+i*3*bottom_size;			
		float* top_=layer_list[layer_id].output_buf+i*3*top_size;
		net_bn_forward( layer_list, layer_id, bottom_, i_c, i_h, i_w,  top_);
		}
}



int 	net_bn_backward(t_layer_info*  layer_list,int layer_id,float* bottom, float* top)
{
	int o_c=layer_list[layer_id].out_dim[0];
	int o_h=layer_list[layer_id].out_dim[1];
	int o_w=layer_list[layer_id].out_dim[2];
	int top_size=o_c*o_h*o_w;
	int bottom_size=top_size;
	float* bottom_diff=bottom+bottom_size;
	float* bottom_diff_temp=bottom+2*bottom_size;
	float* top_diff=top+top_size;
	float* top_data=top;

	float   mean_factor=(float)1.0/(o_h*o_w);
	float* mean_multiplier_=cuda_make_float_array(o_h*o_w);
	float*mean_top_dot_=cuda_make_float_array(o_c);
	cuda_set(o_h*o_w, 1.0, mean_multiplier_);
	float* var_cubic=cuda_make_float_array(o_c*o_h*o_w);

	caffe_gpu_mul(top_size,top_diff,top_data,bottom_diff_temp);		//dim = ch*h*w
	caffe_gpu_gemm(CblasNoTrans,				//step1: cal mean top dot dim=ch*1
		    CblasNoTrans, o_c,1,o_h*o_w,
		    mean_factor, bottom_diff_temp, mean_multiplier_, 0.0,
		    mean_top_dot_);

	caffe_gpu_gemm(CblasNoTrans,				//step2: broadcast mean  dim=ch*h*w
		    CblasNoTrans, o_c,o_h*o_w,1,
		    1.0, mean_top_dot_, mean_multiplier_, 0,
		    bottom_diff_temp);
	caffe_gpu_mul(o_c*o_h*o_w,bottom_diff_temp,top_data,bottom_diff_temp);		//step3: cal mul(yi,channelmean(mul(top_diff,top_data)))		dim=ch*h*w

	caffe_gpu_gemm(CblasNoTrans,				//step3: cal mean top diff dim=ch*1
		    CblasNoTrans, o_c,1,o_h*o_w,
		    mean_factor, top_diff, mean_multiplier_, 0.0,
		    mean_top_dot_);

	caffe_gpu_gemm(CblasNoTrans,				//step4: broadcast mean  dim=ch*h*w
		    CblasNoTrans, o_c,o_h*o_w,1,
		    1.0, mean_top_dot_, mean_multiplier_, 1.0,
		    bottom_diff_temp);

	caffe_gpu_axpby(bottom_size, 1.0, top_diff, -1.0, bottom_diff_temp);

	caffe_gpu_gemm(CblasNoTrans,				//step5: broadcast var dim=ch*h*w
		    CblasNoTrans, o_c,o_h*o_w,1,
		    1.0, layer_list[layer_id].para2_entry, mean_multiplier_, 0.0,
		    var_cubic);

	caffe_gpu_div(o_c*o_h*o_w,bottom_diff_temp,var_cubic,bottom_diff_temp);
	caffe_gpu_axpy(bottom_size, 1.0, bottom_diff_temp,bottom_diff);

	cuda_free(mean_multiplier_);
	cuda_free(var_cubic);
	cuda_free(mean_top_dot_);

	//print_gpu_data(bottom_diff, bottom_size);
	//print_stop_sign("conv back");

	return 1;
}


int net_bn_backward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz)
{
	int bottom_size=layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	for(int i=0;i<batch_sz;i++){
		float* bottom_=layer_list[layer_id].input_buf[0]+i*3*bottom_size;			
		float* top_=layer_list[layer_id].output_buf+i*3*top_size;
		net_bn_backward( layer_list,layer_id, bottom_, top_);
		}
}

