#include "Dropout_layer.hpp"
#include<iostream>
#include<string.h>
#include<float.h>
#include<cmath>
#include<math.h>
#include<vector>
#include<map>
#include"gemm.hpp"
#include"cuda.hpp"

typedef	vector<int>		t_drop_idx;
typedef	map<int, t_drop_idx>	t_drop_map;
static t_drop_map	map_drop;



int 	net_dropout_initial_para(t_layer_info*  layer_list,int layer_id,int in_ch)
{
	layer_list[layer_id].train_inf.para1_size=0;
	layer_list[layer_id].train_inf.para2_size=0;

	map_drop.clear();
	return 0;
}


int 	net_dropout_forward(t_layer_info*  layer_list,int layer_id, float* bottom, int ch, int h, int w,float* top,int batch_item)
{
	int sz=h*w*ch;
	int drop_tag=layer_list[layer_id].layer_name*1000+layer_id+1000000*batch_item;
	float* bottom_data=bottom;
	float* top_data=top;

	float* bottom_host_data=new float[sz];
	cuda_pull_array(bottom_data, bottom_host_data, sz);

#if (TRAIN_MODE>0)
	float* top_host_data = new float[sz];
	float* drop_mask = new float[sz];
	map_drop[drop_tag].clear();
	ZO_fill(0,1, drop_mask, sz);
	for (int i = 0; i < sz; ++i) {
	      	top_host_data[i] = bottom_host_data[i] * drop_mask[i] ;
		if(drop_mask[i]==0){
	       	map_drop[drop_tag].push_back(i);
			}
	    }
	delete []drop_mask;
	delete []top_host_data;
#else
	cuda_push_array(top_data, bottom_host_data, sz);
#endif
	delete []bottom_host_data;
	return 1;
}


int net_dropout_forward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz)
{
	int i_c=layer_list[layer_id].in_dim[0][0];
	int i_h=layer_list[layer_id].in_dim[0][1];
	int i_w=layer_list[layer_id].in_dim[0][2];

	int bottom_size=layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	for(int i=0;i<batch_sz;i++){
		float* bottom_=layer_list[layer_id].input_buf[0]+i*3*bottom_size;			
		float* top_=layer_list[layer_id].output_buf+i*3*top_size;
		net_dropout_forward( layer_list, layer_id, bottom_, i_c, i_h, i_w,  top_,i);
		}
}



int 	net_dropout_backward(t_layer_info*  layer_list,int layer_id,float* bottom, float* top,int batch_item)
{
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	float* top_diff=top+top_size;
	int bottom_size=top_size;
	float* bottom_diff=bottom+bottom_size;
	int drop_tag=layer_list[layer_id].layer_name*1000+layer_id+1000000*batch_item;
	float* bottom_diff_temp=new float[bottom_size];
	float* bottom_diff_host=new float[bottom_size];
	cuda_pull_array(top_diff, bottom_diff_temp, top_size);
	cuda_pull_array(bottom_diff, bottom_diff_host, bottom_size);
	t_drop_idx ll=map_drop[drop_tag];
	for(int i=0;i<ll.size();i++){
		int dropidx=ll[i];
		bottom_diff_temp[dropidx]=0;
		}
	caffe_axpy(bottom_size, 1.0, bottom_diff_temp,bottom_diff_host);
	cuda_push_array(bottom_diff, bottom_diff_host, bottom_size);
	return 1;
}



int net_dropout_backward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz)
{
	int bottom_size=layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	for(int i=0;i<batch_sz;i++){
		float* bottom_=layer_list[layer_id].input_buf[0]+i*3*bottom_size;			
		float* top_=layer_list[layer_id].output_buf+i*3*top_size;
		net_dropout_backward( layer_list,layer_id, bottom_, top_,i);
		}
}



