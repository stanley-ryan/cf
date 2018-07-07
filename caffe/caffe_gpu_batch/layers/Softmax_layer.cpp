#include "Softmax_layer.hpp"
#include <iostream>
#include <string.h>
#include<float.h>
#include<cmath>
#include<math.h>
#include"gemm.hpp"
#include"cuda.hpp"

int 	net_softmax_initial_para(t_layer_info*  layer_list,int layer_id,int in_ch)
{
	layer_list[layer_id].train_inf.para1_size=0;
	layer_list[layer_id].train_inf.para2_size=0;

	return 0;
}


int 	net_softmax_forward(t_layer_info*  layer_list,int layer_id, float* bottom, int ch, int h, int w,float* top)
{
	int i,ic,ih,iw;
	int dim=h*w;
	float* top_data=top;
	int bottom_size=layer_list[layer_id].in_dim[0][2]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][0];
	float* bottom_data=bottom;
	float* bottom_data_host=new float[bottom_size];
	cuda_pull_array(bottom_data, bottom_data_host, bottom_size);

	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	float* top_data_host=new float[top_size];

	float* max_val=new float[dim];
	//float* max_val=(float*)malloc(dim*sizeof(float));
	for(i=0;i<dim;i++){
		max_val[i]=(float)(-999999999);
		}

	for(ic=0;ic<ch;ic++){
		for(i=0;i<dim;i++){
			 max_val[i] = (float)(std::max((float)max_val[i],  (float)bottom_data_host[ic * dim +i]));
			}
		}
	for(ic=0;ic<ch;ic++){
		for(i=0;i<dim;i++){
			 top_data_host[ic*dim+i]= (float)(exp((float)(bottom_data_host[ic * dim +i]- max_val[i])));
			}
		}
	for(i=0;i<dim;i++){
		max_val[i]=0;
		for(ic=0;ic<ch;ic++){
			 max_val[i]+= top_data_host[ic * dim +i];
			}
		}
	for(ic=0;ic<ch;ic++){
		for(i=0;i<dim;i++){
			top_data_host[ic*dim+i]=(float)(top_data_host[ic * dim +i]/max_val[i]);
			}
		}

	cuda_push_array(top_data, top_data_host, top_size);
	//print_gpu_data(top_data, top_size);
	delete [] max_val;
	delete []bottom_data_host;
	delete []top_data_host;

	return 1;
}


int net_softmax_forward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz)
{
	int i_c=layer_list[layer_id].in_dim[0][0];
	int i_h=layer_list[layer_id].in_dim[0][1];
	int i_w=layer_list[layer_id].in_dim[0][2];

	int bottom_size=layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	for(int i=0;i<batch_sz;i++){
		float* bottom_=layer_list[layer_id].input_buf[0]+i*3*bottom_size;			
		float* top_=layer_list[layer_id].output_buf+i*3*top_size;
		net_softmax_forward( layer_list, layer_id, bottom_, i_c, i_h, i_w,  top_);
		}
}




int 	net_softmax_backward(t_layer_info*  layer_list,int layer_id)
{

	return 1;
}


