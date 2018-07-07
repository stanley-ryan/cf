#include "Prelu_layer.hpp"
#include <iostream>
#include <string.h>
#include<pthread.h>
#include"gemm.hpp"
#include"cuda.hpp"


int 	net_prelu_initial_para(t_layer_info*  layer_list,int layer_id,int in_ch)
{
	float rnd_val;
	int prelu_w_size=layer_list[layer_id].output_num;

	float* prelu_w =new float[prelu_w_size];
	if(layer_list[layer_id].train_inf.para1_ini_way==para_ini_loading){
		layer_list[layer_id].para1_entry=cuda_make_array(layer_list[layer_id].para1_entry,prelu_w_size);
		printf("initial para1 prelu_layer_%d: loading parameter\n",layer_id);
	}else{
		for(int i=0;i<prelu_w_size;i++){
			prelu_w[i]=0.25;
			}
		layer_list[layer_id].para1_entry=cuda_make_array(prelu_w,prelu_w_size);
		printf("initial para1 prelu_layer_%d: initial parameter\n",layer_id);
	}
	delete []prelu_w;

	if(layer_list[layer_id].train_inf.para_diff_way==e_para_unchange){
		printf("layer id =%d, all zero for para1/2_diff\n",layer_id);
		}


	layer_list[layer_id].train_inf.para1_diff=cuda_make_float_array(prelu_w_size);
	layer_list[layer_id].train_inf.para1_diff_hist=cuda_make_float_array(prelu_w_size);
	layer_list[layer_id].train_inf.para1_size=prelu_w_size;
	layer_list[layer_id].train_inf.para2_size=0;
	cuda_set(prelu_w_size,0, layer_list[layer_id].train_inf.para1_diff);
	cuda_set(prelu_w_size,0, layer_list[layer_id].train_inf.para1_diff_hist);

	return 0;
}



extern void PreLULayer_Forward_gpu(const int count, const int channels, const int dim,
									const float* bottom_data, float* top_data, const float* slope_data,
									const int div_factor, int num_kernels);


int 	net_prelu_forward(t_layer_info*  layer_list,int layer_id, float* bottom, int ch, int h, int w,float* top)
{
	int i;
	int dim=h*w;
	int channels=ch;
	int count=dim*ch;
	float* slope_data=layer_list[layer_id].para1_entry;
	int div_factor = (prelu_share_channel==layer_list[layer_id].ker_nfo.act) ? channels : 1;
	float* bottom_data=bottom;

  	PreLULayer_Forward_gpu(count, channels, dim, bottom_data, top, slope_data, div_factor, count);

	return 1;
}


int net_prelu_forward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz)
{
	int i_c=layer_list[layer_id].in_dim[0][0];
	int i_h=layer_list[layer_id].in_dim[0][1];
	int i_w=layer_list[layer_id].in_dim[0][2];

	int bottom_size=layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	for(int i=0;i<batch_sz;i++){
		float* bottom_=layer_list[layer_id].input_buf[0]+i*3*bottom_size;
		float* top_=layer_list[layer_id].output_buf+i*3*top_size;
		net_prelu_forward( layer_list, layer_id, bottom_, i_c, i_h, i_w,  top_);
		}
}




extern void PreLULayer_Backward_gpu(const int count, const int channels, const int dim,
										const float* bottom_data, float* bottom_diff, float* top_diff, float* slope_diff, const float* slope_data,
										const int div_factor, int num_kernels);

int 	net_prelu_backward(t_layer_info*  layer_list,int layer_id,float* bottom, float* top)
{
	int bottom_size=layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int dim=layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int top_size=bottom_size;
	int i;
	int count=bottom_size;
	int channels=layer_list[layer_id].in_dim[0][0];
	int div_factor = (prelu_share_channel==layer_list[layer_id].ker_nfo.act) ? channels : 1;
	float* slope_data=layer_list[layer_id].para1_entry;

	float* slope_diff=layer_list[layer_id].train_inf.para1_diff;
	float* bottom_diff=bottom+bottom_size;
	float* bottom_data=bottom;
	float* top_diff=top+top_size;
	float* bottom_diff_temp=bottom+bottom_size*2;


	PreLULayer_Backward_gpu(count, channels, dim, bottom_data, bottom_diff_temp, top_diff, slope_diff, slope_data, div_factor,count);
	caffe_gpu_axpy(bottom_size, 1.0, bottom_diff_temp,bottom_diff);

	if(layer_list[layer_id].train_inf.para_diff_way==e_para_unchange){
		cuda_set(channels,0,layer_list[layer_id].train_inf.para1_diff);
		}

	return 1;
}


int net_prelu_backward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz)
{
	int bottom_size=layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	for(int i=0;i<batch_sz;i++){
		float* bottom_=layer_list[layer_id].input_buf[0]+i*3*bottom_size;
		float* top_=layer_list[layer_id].output_buf+i*3*top_size;
		net_prelu_backward( layer_list,layer_id, bottom_, top_);
		}
}

