#include "Eltwise_layer.hpp"
#include <iostream>
#include <string.h>
#include<pthread.h>
#include "gemm.hpp"
#include"cuda.hpp"


typedef struct eltwise_thread_para{
	t_layer_info*  layer_list;
	int layer_id;
	float* input1;
	float* input2;
	int ch;
	int h;
	int w;
	float* output;
	 int start_r;
	 int end_r;
}t_eltwise_thread_para;


void* Eltwise_sum_thread(void* args)
{
	t_eltwise_thread_para*  thread_para=(t_eltwise_thread_para* )args;
	t_layer_info*  layer_list=thread_para->layer_list;
	int layer_id=thread_para->layer_id;
	float* input1=thread_para->input1;
	float* input2=thread_para->input2;
	int h=thread_para->h;
	int w=thread_para->w;
	int ch=thread_para->ch;
	float* output=thread_para->output;
	 int start_r=thread_para->start_r;
	 int end_r=thread_para->end_r;
	int i;
	int dim=h*w;
	int channels=ch;
	int count=dim*ch;
	float* mult=layer_list[layer_id].para1_entry;

	for(i = start_r; i < end_r; i++) {
	   	output[i] =(float)mult[0]*input1[i]  + (float)mult[1]*input2[i];
	  	}

}




int 	net_eltwise_initial_para(t_layer_info*  layer_list,int layer_id,int in_ch)
{
	layer_list[layer_id].train_inf.para1_size=get_bottom_num(layer_list, layer_id);
	layer_list[layer_id].train_inf.para1_bp_mode=e_para_unbp;
	layer_list[layer_id].train_inf.para2_size=0;
	printf("eltwise layer[%d],input num=%d\n",layer_id,layer_list[layer_id].train_inf.para1_size);
	return 0;
}


int 	net_Eltwise_forward(t_layer_info*  layer_list,int layer_id, int ch, int h, int w,float* top,float** bottom_ptr)
{
	int i;
	int dim=h*w;
	int channels=ch;
	int count=dim*ch;
	float* mult=layer_list[layer_id].para1_entry;

	//if(MULT_THREAD<=1){
		if(layer_list[layer_id].ker_nfo.act==eltw_sum){
    			caffe_gpu_set(count, 0.0, top);    // TODO(shelhamer) does cuBLAS optimize to sum for coeff = 1?
    			for (int i = 0; i < get_bottom_num(layer_list, layer_id); i++) {
				float* input=bottom_ptr[i];
				caffe_gpu_axpy(count, mult[i], input, top);
				}
		}else{
			printf("unsuported mode for eltwise layer\n");
			exit(-1);
			}
	return 1;
}




int net_Eltwise_forward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz)
{
	int i_c=layer_list[layer_id].in_dim[0][0];
	int i_h=layer_list[layer_id].in_dim[0][1];
	int i_w=layer_list[layer_id].in_dim[0][2];				
	int bottom_size=i_c*i_h*i_w;
	float* bottom_[MAX_IN];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	
	for(int i=0;i<batch_sz;i++){
		for(int j=0;j<get_bottom_num(layer_list, layer_id);j++){
			bottom_[j]=layer_list[layer_id].input_buf[j]+i*3*bottom_size;	
			}
		float* top_=layer_list[layer_id].output_buf+i*3*top_size;
		net_Eltwise_forward( layer_list, layer_id, i_c, i_h, i_w,  top_,bottom_);
		}
}




int 	net_Eltwise_backward(t_layer_info*  layer_list,int layer_id,float* top,float**  bottom_ptr)
{

	int i;
	float* mult=layer_list[layer_id].para1_entry;

	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	float* top_diff=top+top_size;

	int bottom_size[MAX_IN];
	float* bottom_diff[MAX_IN];
	float* bottom_diff_temp[MAX_IN];
	for(i=0;i<get_bottom_num(layer_list, layer_id);i++){
		bottom_size[i]=layer_list[layer_id].in_dim[i][0]*layer_list[layer_id].in_dim[i][1]*layer_list[layer_id].in_dim[i][2];
		bottom_diff[i]=bottom_ptr[i]+bottom_size[i];
		bottom_diff_temp[i]=bottom_ptr[i]+2*bottom_size[i];
		}

	if(layer_list[layer_id].ker_nfo.act==eltw_sum){
	    	for (int i = 0; i < get_bottom_num(layer_list, layer_id); i++) {
			if (mult[i] == float(1.)) {
				caffe_gpu_copy(bottom_size[0], top_diff, bottom_diff_temp[i]);
			} else {
				caffe_gpu_copy(bottom_size[0], top_diff, bottom_diff_temp[i]);
				caffe_gpu_scal(bottom_size[0], mult[i], bottom_diff_temp[i]);
				}
			caffe_gpu_axpy(bottom_size[i], 1.0, bottom_diff_temp[i],bottom_diff[i]);
			}
	}else{
		printf("unsuported mode for eltwise layer\n");
		exit(-1);
		}

	return 1;
}



int net_Eltwise_backward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz)
{
	int bottom_size=layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	float* bottom_[MAX_IN];

	for(int i=0;i<batch_sz;i++){
		for(int j=0;j<get_bottom_num(layer_list, layer_id);j++){
			bottom_[j]=layer_list[layer_id].input_buf[j]+i*3*bottom_size;	
			}
		float* top_=layer_list[layer_id].output_buf+i*3*top_size;
		net_Eltwise_backward( layer_list,layer_id,  top_,bottom_);
		}
}

