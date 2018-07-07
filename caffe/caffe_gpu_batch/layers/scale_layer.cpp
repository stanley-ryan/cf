
#include <iostream>
#include <string.h>

#include <stdio.h>
#include <getopt.h>
#include <sys/time.h>
#include <time.h>
#include<pthread.h>
#include "gemm.hpp"
#include "scale_layer.hpp"
#include"cuda.hpp"


int 	net_scale_initial_para(t_layer_info*  layer_list,int layer_id,int in_ch)
{
	int para_size;
	int o_c=layer_list[layer_id].output_num;
	para_size=o_c;

	if(layer_list[layer_id].train_inf.para1_ini_way==para_ini_loading){		//scale item
		layer_list[layer_id].para1_entry=cuda_make_array(layer_list[layer_id].para1_entry,para_size);
		printf("initial para1 scale_layer_%d: loading parameter\n",layer_id);
	}else{
		layer_list[layer_id].para1_entry=cuda_make_float_array(para_size);
		cuda_set(para_size, 1.0, layer_list[layer_id].para1_entry);
		}

	if(layer_list[layer_id].train_inf.en_bias){
		if(layer_list[layer_id].train_inf.para2_ini_way==para_ini_loading){		//bias item
			layer_list[layer_id].para2_entry=cuda_make_array(layer_list[layer_id].para2_entry,para_size);
			printf("initial para2 scale_layer_%d: loading parameter\n",layer_id);
		}else{
			layer_list[layer_id].para2_entry=cuda_make_float_array(para_size);
			cuda_set(para_size, 0.0, layer_list[layer_id].para2_entry);
			}
		}

	layer_list[layer_id].train_inf.para1_diff=cuda_make_float_array(para_size);
	layer_list[layer_id].train_inf.para1_diff_hist=cuda_make_float_array(para_size);
	layer_list[layer_id].train_inf.para1_size=para_size;
	if(layer_list[layer_id].train_inf.en_bias){
		layer_list[layer_id].train_inf.para2_diff=cuda_make_float_array(para_size);
		layer_list[layer_id].train_inf.para2_diff_hist=cuda_make_float_array(para_size);
		layer_list[layer_id].train_inf.para2_size=para_size;
		}
	cuda_set(para_size,0, layer_list[layer_id].train_inf.para1_diff);
	cuda_set(para_size,0, layer_list[layer_id].train_inf.para1_diff_hist);
	if(layer_list[layer_id].train_inf.en_bias){
		cuda_set(para_size,0, layer_list[layer_id].train_inf.para2_diff);
		cuda_set(para_size,0, layer_list[layer_id].train_inf.para2_diff_hist);
		}

	return 0;
}




int 	net_scale_forward(t_layer_info*  layer_list,int layer_id, float* bottom, int ch, int h, int w,float* top)
{
	int o_h;
	int o_w;
	int o_c;
	float* bottom_data=bottom;
	float* top_data=top;

	o_c=layer_list[layer_id].out_dim[0];
	o_h=layer_list[layer_id].out_dim[1];
	o_w=layer_list[layer_id].out_dim[2];

	float* mean_multiplier_=cuda_make_float_array(o_h*o_w);
	cuda_set(o_h*o_w, 1.0, mean_multiplier_);

	caffe_gpu_gemm(CblasNoTrans,				//step2: broadcast scale dim=ch*h*w
			CblasNoTrans, o_c,o_h*o_w,1,
			1.0, layer_list[layer_id].para1_entry, mean_multiplier_, 0.0,
			top_data);

	caffe_gpu_mul(o_c*o_h*o_w,top_data,bottom_data,top_data);		//step3: mul scale*x		dim=ch*h*w

	if(layer_list[layer_id].train_inf.en_bias){
		caffe_gpu_gemm(CblasNoTrans,				//step4: broadcast bias and add on scale*x dim=ch*h*w
			    CblasNoTrans, o_c,o_h*o_w,1,
			    1.0, layer_list[layer_id].para2_entry, mean_multiplier_, 1.0,
			    top_data);
		}
	cuda_free(mean_multiplier_);

	return 1;
}


int net_scale_forward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz)
{
	int i_c=layer_list[layer_id].in_dim[0][0];
	int i_h=layer_list[layer_id].in_dim[0][1];
	int i_w=layer_list[layer_id].in_dim[0][2];

	int bottom_size=layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	for(int i=0;i<batch_sz;i++){
		float* bottom_=layer_list[layer_id].input_buf[0]+i*3*bottom_size;			
		float* top_=layer_list[layer_id].output_buf+i*3*top_size;
		net_scale_forward( layer_list, layer_id, bottom_, i_c, i_h, i_w,  top_);
		}
}



int 	net_scale_backward(t_layer_info*  layer_list,int layer_id,float * bottom, float* top)
{
	int o_c=layer_list[layer_id].out_dim[0];
	int o_h=layer_list[layer_id].out_dim[1];
	int o_w=layer_list[layer_id].out_dim[2];
	int top_size=o_c*o_h*o_w;
	int bottom_size=top_size;
	float* bottom_data=bottom;
	float* bottom_diff=bottom+bottom_size;
	float* bottom_diff_temp=bottom+2*bottom_size;
	float* top_diff=top+top_size;
	float* top_data=top;

	float* mean_multiplier_=cuda_make_float_array(o_h*o_w);
	cuda_set(o_h*o_w, 1.0, mean_multiplier_);

	caffe_gpu_gemm(CblasNoTrans,				//step2: broadcast scale dim=ch*h*w
			CblasNoTrans, o_c,o_h*o_w,1,
			1.0, layer_list[layer_id].para1_entry, mean_multiplier_, 0.0,
			bottom_diff_temp);

	caffe_gpu_mul(bottom_size,top_diff,bottom_diff_temp,bottom_diff_temp);		//step3: mul scale*x		dim=ch*h*w
	caffe_gpu_axpy(bottom_size, 1.0, bottom_diff_temp,bottom_diff);			//add on bottom diff

	caffe_gpu_mul(bottom_size,top_diff,bottom_data,bottom_diff_temp);
	caffe_gpu_gemm(CblasNoTrans,							//step1: cal mean top dot dim=ch*1
		    CblasNoTrans, o_c,1,o_h*o_w,
		    1.0, bottom_diff_temp, mean_multiplier_, 1.0,
		    layer_list[layer_id].train_inf.para1_diff);

	if(layer_list[layer_id].train_inf.en_bias && (layer_list[layer_id].train_inf.para2_bp_mode==e_para_bp)){
		caffe_gpu_gemm(CblasNoTrans,							//step1: cal mean top dot dim=ch*1
			    CblasNoTrans, o_c,1,o_h*o_w,
			    1.0, top_diff, mean_multiplier_, 1.0,
			    layer_list[layer_id].train_inf.para2_diff);
		}

	cuda_free(mean_multiplier_);

	return 1;
}


int net_scale_backward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz)
{
	int bottom_size=layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	for(int i=0;i<batch_sz;i++){
		float* bottom_=layer_list[layer_id].input_buf[0]+i*3*bottom_size;			
		float* top_=layer_list[layer_id].output_buf+i*3*top_size;
		net_scale_backward( layer_list,layer_id, bottom_, top_);
		}
}

