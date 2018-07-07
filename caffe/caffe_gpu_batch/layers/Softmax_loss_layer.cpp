#include "Softmax_loss_layer.hpp"
#include <iostream>
#include <string.h>
#include<float.h>
#include<cmath>
#include<math.h>

#include"gemm.hpp"
#include"cuda.hpp"

int 	net_softmax_loss_initial_para(t_layer_info*  layer_list,int layer_id,int in_ch)
{
	layer_list[layer_id].train_inf.para1_size=0;
	layer_list[layer_id].train_inf.para2_size=1;
	layer_list[layer_id].train_inf.para2_bp_mode=e_para_unbp;

	return 0;
}


int 	net_softmax_loss_forward(t_layer_info*  layer_list,int layer_id, float* bottom, int ch, int h, int w,float* top, t_label  label)
{
	float loss=0;
	int bottom_size=layer_list[layer_id].in_dim[0][2]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][0];
	float* bottom_data=bottom;
	float* bottom_diff=bottom+bottom_size;
	float* top_data=top;
	float* prob_data=bottom+bottom_size*2;
	float softmax_loss_weight=layer_list[layer_id].para2_entry[0];

	float* bottom_data_host=new float[bottom_size];
	float* prob_data_host=new float[bottom_size];

	//printf("sisisisisi bottom_size=%d\n",bottom_size);
	
	cuda_pull_array(bottom_data, bottom_data_host, bottom_size);
	//printf("oooooo\n");


	float sum_exp=0;

	for(int i=0;i<bottom_size;i++){
		prob_data_host[i]=exp(bottom_data_host[i]);
		sum_exp+=prob_data_host[i];
		}
	for(int i=0;i<bottom_size;i++){
		prob_data_host[i]=prob_data_host[i]/sum_exp;
		}
	if(label!=EMPTY_LABEL){
		loss-=log(prob_data_host[label]);
		top_data[0]=loss*softmax_loss_weight;
		}

	int most_like_label=0;
	float most_probdata=prob_data_host[0];
	for(int i=1;i<bottom_size;i++){
		if(most_probdata<prob_data_host[i]){
			most_probdata=prob_data_host[i];
			most_like_label=i;
			}
		}
	//printf("kekek\n");

	top_data[1]=most_like_label;
	top_data[2]=most_probdata;
	//printf("lelelel\n");

	//cuda_push_array(bottom_data, bottom_data_host, bottom_size);
	cuda_push_array(prob_data, prob_data_host, bottom_size);
	//printf("hahahahah\n");



#if 0
	if(label!=EMPTY_LABEL){
		printf("softmax loss=%f, label=%d, prob=%f\n",top_data[0], most_like_label, most_probdata);
	}else{
		printf(" label=%d, prob=%f\n", most_like_label, most_probdata);
		}
	//getchar();
#endif

	delete [] bottom_data_host;
	delete [] prob_data_host;

	return 1;
}


int net_softmax_loss_forward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz,t_label* label_list)
{
	int i_c=layer_list[layer_id].in_dim[0][0];
	int i_h=layer_list[layer_id].in_dim[0][1];
	int i_w=layer_list[layer_id].in_dim[0][2];
	//printf("ic=%d,i_h=%d,i_w=%d\n",i_c,i_h,i_w);
	int bottom_size=layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	for(int i=0;i<batch_sz;i++){
		float* bottom_=layer_list[layer_id].input_buf[0]+i*3*bottom_size;			
		float* top_=layer_list[layer_id].output_buf+i*3*top_size;
		//if(batch_sz>=2){
		//	print_gpu_data(bottom_,bottom_size);
		//	print_stop_sign("softmaxloss_layer\n");
		//	}
		net_softmax_loss_forward( layer_list, layer_id, bottom_, i_c, i_h, i_w,  top_,label_list[i]);
		}
}



int 	net_softmax_loss_backward(t_layer_info*  layer_list,int layer_id, t_label label, float* bottom, float* top)
{
	float loss=0;
	int bottom_size=layer_list[layer_id].in_dim[0][2]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][0];
	float* bottom_data=bottom;
	float* bottom_diff=bottom+bottom_size;
	float* top_data=top;
	float softmax_loss_weight=layer_list[layer_id].para2_entry[0];
	float* prob_data=bottom+bottom_size*2;

	float* prob_data_host=new float[bottom_size];
	float* bottom_diff_host=new float[bottom_size];

	cuda_pull_array(prob_data, prob_data_host, bottom_size);
	cuda_pull_array(bottom_diff, bottom_diff_host, bottom_size);

	prob_data_host[label] = prob_data_host[label] - 1;
	caffe_scal(bottom_size, softmax_loss_weight, prob_data_host);
	caffe_axpy(bottom_size, 1.0, prob_data_host,bottom_diff_host);

	cuda_push_array(bottom_diff, bottom_diff_host, bottom_size);
	//cuda_push_array(prob_data, prob_data_host, bottom_size);

	delete [] prob_data_host;
	delete [] bottom_diff_host;

	return 1;
}

int net_softmax_loss_backward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz,t_label* label_list)
{
	int bottom_size=layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	for(int i=0;i<batch_sz;i++){
		float* bottom_=layer_list[layer_id].input_buf[0]+i*3*bottom_size;			
		float* top_=layer_list[layer_id].output_buf+i*3*top_size;
		net_softmax_loss_backward( layer_list, layer_id, label_list[i], bottom_, top_);
		}
}

