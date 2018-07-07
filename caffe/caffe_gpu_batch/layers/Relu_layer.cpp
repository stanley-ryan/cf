#include "Relu_layer.hpp"
#include <iostream>
#include <string.h>
#include<pthread.h>
#include"cuda.hpp"
#include "gemm.hpp"



typedef struct relu_thread_para{
	t_layer_info*  layer_list;
	int layer_id;
	float* input;
	int ch;
	int h;
	int w;
	float* output;
	 int start_r;
	 int end_r;
}t_relu_thread_para;





int 	net_relu_initial_para(t_layer_info*  layer_list,int layer_id,int in_ch)
{
	layer_list[layer_id].train_inf.para1_size=1;
	layer_list[layer_id].train_inf.para1_bp_mode=e_para_unbp;
	layer_list[layer_id].train_inf.para2_size=0;

	return 0;
}


extern	void ReLULayer_Forward_gpu(float negative_slope, float* bottom_data, float* top_data, int count);

int 	net_relu_forward(t_layer_info*  layer_list,int layer_id, float* input, int ch, int h, int w,float* output)
{
	int i;
	int dim=h*w;
	int channels=ch;
	int count=dim*ch;
	float* slope_data=layer_list[layer_id].para1_entry;

	ReLULayer_Forward_gpu(slope_data[0], input, output,  count);

	//print_gpu_data(output , count);

	return 1;
}


int net_relu_forward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz)
{
	int i_c=layer_list[layer_id].in_dim[0][0];
	int i_h=layer_list[layer_id].in_dim[0][1];
	int i_w=layer_list[layer_id].in_dim[0][2];

	int bottom_size=layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	for(int i=0;i<batch_sz;i++){
		float* bottom_=layer_list[layer_id].input_buf[0]+i*3*bottom_size;
		float* top_=layer_list[layer_id].output_buf+i*3*top_size;
		net_relu_forward( layer_list, layer_id, bottom_, i_c, i_h, i_w,  top_);
		}
}




extern	void ReLULayer_Backward_gpu(float negative_slope, float* top_diff, float* bottom_data, float* bottom_diff,int count);

int 	net_relu_backward(t_layer_info*  layer_list,int layer_id,float* bottom, float* top)
{
	float* slope_data=layer_list[layer_id].para1_entry;

	int bottom_size=layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int top_size=bottom_size;
	float* bottom_diff_temp=bottom+bottom_size*2;
	float* bottom_diff=bottom+bottom_size;
	float* bottom_data=bottom;
	float* top_diff=top+top_size;

	ReLULayer_Backward_gpu(slope_data[0], top_diff, bottom_data, bottom_diff_temp,bottom_size);

	caffe_gpu_axpy(bottom_size, 1.0, bottom_diff_temp,bottom_diff);

	return 1;
}



int net_relu_backward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz)
{
	int bottom_size=layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	for(int i=0;i<batch_sz;i++){
		float* bottom_=layer_list[layer_id].input_buf[0]+i*3*bottom_size;
		float* top_=layer_list[layer_id].output_buf+i*3*top_size;
		net_relu_backward( layer_list,layer_id, bottom_, top_);
		}
}


