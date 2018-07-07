#include "Concat_layer.hpp"
#include <iostream>
#include <string.h>
#include<pthread.h>

int 	net_concat_initial_para(t_layer_info*  layer_list,int layer_id,int in_ch)
{
	layer_list[layer_id].train_inf.para1_size=0;
	layer_list[layer_id].train_inf.para2_size=0;

	return 0;
}


int 	net_Concat_forward(t_layer_info*  layer_list,int layer_id, float* input1,float* input2 ,float* input3,float* input4 ,float* output)
{
	int i;
	int bottom_size[MAX_IN];
	float* bottom_data[MAX_IN];
	int top_size[MAX_IN];
	float* top_data[MAX_IN];

	for(i=0;i<MAX_IN;i++){
		bottom_size[i]=layer_list[layer_id].in_dim[i][0]*layer_list[layer_id].in_dim[i][1]*layer_list[layer_id].in_dim[i][2];
		bottom_data[i]=layer_list[layer_id].input_buf[i];
		top_size[i]=bottom_size[i];
		}

	top_data[0]=layer_list[layer_id].output_buf;
	memcpy((void*)top_data[0],(void*)bottom_data[0],bottom_size[0]*sizeof(float));
	for(i=1;i<MAX_IN;i++){
		top_data[i]=top_data[i-1]+top_size[i-1];
		memcpy((void*)top_data[i],(void*)bottom_data[i],top_size[i]*sizeof(float));
		}

	return 1;
}



int 	net_concat_backward(t_layer_info*  layer_list,int layer_id)
{
	int i;
	int bottom_size[MAX_IN];
	float* bottom_diff[MAX_IN];
	int top_size[MAX_IN];
	float* top_diff[MAX_IN];
	int total_top_size=0;

	for(i=0;i<MAX_IN;i++){
		bottom_size[i]=layer_list[layer_id].in_dim[i][0]*layer_list[layer_id].in_dim[i][1]*layer_list[layer_id].in_dim[i][2];
		bottom_diff[i]=layer_list[layer_id].input_buf[i]+bottom_size[i];
		top_size[i]=bottom_size[i];
		total_top_size+=top_size[i];
		}

	top_diff[0]=layer_list[layer_id].output_buf+total_top_size;
	memcpy((void*)bottom_diff[0],(void*)top_diff[0],top_size[0]*sizeof(float));
	for(i=1;i<MAX_IN;i++){
		top_diff[i]=top_diff[i-1]+top_size[i-1];
		memcpy((void*)bottom_diff[i],(void*)top_diff[i],top_size[i]*sizeof(float));
		}

	return 1;
}

