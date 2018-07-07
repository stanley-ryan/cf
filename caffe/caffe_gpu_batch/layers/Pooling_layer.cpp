#include "Pooling_layer.hpp"
#include <iostream>
#include <string.h>
#include<float.h>
#include <map>
#include"gemm.hpp"
#include"cuda.hpp"

using namespace std;

typedef	map<int, int*>	t_pool_map;
static t_pool_map	map_out_in;

int 	net_pool_initial_para(t_layer_info*  layer_list,int layer_id,int in_ch)
{
	layer_list[layer_id].train_inf.para1_size=0;
	layer_list[layer_id].train_inf.para2_size=0;

	map_out_in.clear();
	return 0;
}


extern void PoolingLayer_Forward_gpu(t_layer_info*  layer_list,int layer_id, float* input, int ch, int h, int w,float* output, int *mask);

int 	net_pool_forward(t_layer_info*  layer_list,int layer_id, float* input, int ch, int h, int w,float* output,int batch_item)
{
	int channels_=ch;
	int pooled_height_=layer_list[layer_id].out_dim[1];
	int pooled_width_=layer_list[layer_id].out_dim[2];
	int num_kernels = pooled_height_*pooled_width_*channels_;
	int bottom_size=layer_list[layer_id].in_dim[0][2]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][0];

	int pool_tag=layer_list[layer_id].layer_name*1000+layer_id+1000000*batch_item;
	int* mask= map_out_in[pool_tag];
	if(mask==NULL){
		map_out_in[pool_tag]=cuda_make_int_array(bottom_size);
		mask= map_out_in[pool_tag];
	}
	cuda_set_int(num_kernels,0,mask);
	PoolingLayer_Forward_gpu(layer_list,layer_id, input, ch, h, w, output, mask);

	//print_gpu_data(output,num_kernels);

	return 0;
}


int net_pool_forward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz)
{
	int i_c=layer_list[layer_id].in_dim[0][0];
	int i_h=layer_list[layer_id].in_dim[0][1];
	int i_w=layer_list[layer_id].in_dim[0][2];

	int bottom_size=layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	for(int i=0;i<batch_sz;i++){
		float* bottom_=layer_list[layer_id].input_buf[0]+i*3*bottom_size;			
		float* top_=layer_list[layer_id].output_buf+i*3*top_size;
		net_pool_forward( layer_list, layer_id, bottom_, i_c, i_h, i_w,  top_,i);
		}
}




extern	void PoolingLayer_Backward_gpu(t_layer_info*  layer_list,int layer_id, int *mask,float* bottom, float* top);

int 	net_pool_backward(t_layer_info*  layer_list,int layer_id,float* bottom, float* top,int batch_item)
{
	int bottom_size=layer_list[layer_id].in_dim[0][2]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][0];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	float* bottom_diff=bottom+bottom_size;
	float* top_diff=top+top_size;
	int pool_tag=layer_list[layer_id].layer_name*1000+layer_id+1000000*batch_item;
	int* mask= map_out_in[pool_tag];

	PoolingLayer_Backward_gpu(layer_list, layer_id, mask,bottom, top);

#if 0
	printf("layer_id=%d\n",layer_id);
	getchar();
	print_gpu_data(bottom_diff, bottom_size);
	print_stop_sign("pool back");
	//cuda_free(map_out_in[pool_tag]);
#endif
}



int net_pool_backward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz)
{
	int bottom_size=layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	for(int i=0;i<batch_sz;i++){
		float* bottom_=layer_list[layer_id].input_buf[0]+i*3*bottom_size;
		float* top_=layer_list[layer_id].output_buf+i*3*top_size;
		net_pool_backward( layer_list,layer_id, bottom_, top_,i);
		}
}

