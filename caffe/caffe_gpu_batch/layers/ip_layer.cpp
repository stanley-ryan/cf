#include "ip_layer.hpp"
#include <iostream>
#include <string.h>
#include<pthread.h>

#include "gemm.hpp"
#include"cuda.hpp"

typedef struct ip_thread_para{
	t_layer_info*  layer_list;
	int layer_id;
	float* input;
	int in_ch;
	float* output;
	 int start_r;
	 int end_r;
}t_ip_thread_para;



static void inner_product_without_sparse(t_layer_info*  layer_list,int layer_id,float* input, int in_ch, float* output)
{
	int out_ch=layer_list[layer_id].output_num;
	float* para_w=layer_list[layer_id].para1_entry;
	float* para_b=layer_list[layer_id].para2_entry;

	int oi,ii,pi;
	int i;

	caffe_gpu_gemm(CblasNoTrans, CblasNoTrans,out_ch,1, in_ch, 1, para_w,input, 0, output);
	if(layer_list[layer_id].train_inf.en_bias){
		float* bias_multiplier_=cuda_make_float_array(1);
		cuda_set(1,1.0,bias_multiplier_);
		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, out_ch, 1, 1, 1., para_b, bias_multiplier_,	1., output);
		cuda_free(bias_multiplier_);
		}
}

int 	net_ip_initial_para(t_layer_info*  layer_list,int layer_id,int in_ch)
{
	int para_size;
	int ip_w_size=layer_list[layer_id].output_num*layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int fan_in=in_ch;
	int fan_out=layer_list[layer_id].output_num;

	para_size=ip_w_size;
	float* ip_w =new float[para_size];
	if(layer_list[layer_id].train_inf.para1_ini_way==para_ini_gaussian){
		gaussian_fill(0.01,0.0,ip_w, para_size);
		layer_list[layer_id].para1_entry=cuda_make_array(ip_w,para_size);
		printf("initial para1 ip_layer_%d: initial parameter gaussian\n",layer_id);
	}else if(layer_list[layer_id].train_inf.para1_ini_way==para_ini_xavier){
		float scale=sqrt(3.0/fan_in);
		unirom_fill(-scale, scale, ip_w, para_size);
		layer_list[layer_id].para1_entry=cuda_make_array(ip_w,para_size);
		printf("initial para1 ip_layer_%d: initial parameter uninorm\n",layer_id);
	}else if(layer_list[layer_id].train_inf.para1_ini_way==para_ini_loading){
		layer_list[layer_id].para1_entry=cuda_make_array(layer_list[layer_id].para1_entry,para_size);
		printf("initial para1 ip_layer_%d: loading parameter\n",layer_id);
	}else{
		printf("initial para1 ip_layer_%d: invalid parameter initial way\n",layer_id);
		exit(-1);
		}
	delete [] ip_w;

	if(layer_list[layer_id].train_inf.en_bias){
		para_size=layer_list[layer_id].output_num;
		fan_in=in_ch;
		float* ip_b =new float[para_size];
		if(layer_list[layer_id].train_inf.para2_ini_way==para_ini_gaussian){
			gaussian_fill(0.01,0.0,ip_b, para_size);
			layer_list[layer_id].para2_entry=cuda_make_array(ip_b,para_size);
			printf("initial para2 ip_layer_%d: initial parameter gaussian\n",layer_id);
		}else if(layer_list[layer_id].train_inf.para2_ini_way==para_ini_xavier){
			float scale=sqrt(3.0/fan_in);
			unirom_fill(-scale, scale, ip_b, para_size);
			layer_list[layer_id].para2_entry=cuda_make_array(ip_b,para_size);
			printf("initial para2 ip_layer_%d: initial parameter uninorm\n",layer_id);
		}else if(layer_list[layer_id].train_inf.para2_ini_way==para_ini_loading){
			layer_list[layer_id].para2_entry=cuda_make_array(layer_list[layer_id].para2_entry,para_size);
			printf("initial para2 ip_layer_%d: loading parameter\n",layer_id);
		}else if(layer_list[layer_id].train_inf.para2_ini_way==para_ini_const_zero){
			layer_list[layer_id].para2_entry=cuda_make_float_array(para_size);
			cuda_set(para_size, 0.0, layer_list[layer_id].para2_entry);
		}else{
			printf("initial para2 ip_layer_%d: invalid parameter initial way\n",layer_id);
			exit(-1);
			}
		delete [] ip_b;
		}

	if(layer_list[layer_id].train_inf.para_diff_way==e_para1_prune){
		layer_list[layer_id].train_inf.para_diff_mask=new float[ip_w_size];
		int channel_size=ip_w_size/layer_list[layer_id].output_num;
		float* ch_ptr=layer_list[layer_id].train_inf.para_diff_mask;
		for(int i=0;i<layer_list[layer_id].output_num;i++){
			rnd_mask(channel_size, ch_ptr, layer_list[layer_id].train_inf.prune_percent,0,1);
			ch_ptr+=channel_size;
			}
		float* para1_data_host=new float[ip_w_size];
		cuda_pull_array(layer_list[layer_id].para1_entry, para1_data_host, ip_w_size);
		caffe_mul(ip_w_size, para1_data_host, layer_list[layer_id].train_inf.para_diff_mask, para1_data_host);
		cuda_push_array(layer_list[layer_id].para1_entry, para1_data_host, ip_w_size);
		delete []para1_data_host;
	}else if(layer_list[layer_id].train_inf.para_diff_way==e_para_unchange){
		//layer_list[layer_id].train_inf.para_diff_mask=new float[conv_w_size];
		//caffe_set(conv_w_size,0,layer_list[layer_id].train_inf.para_diff_mask);
		printf("layer id =%d, all zero for para1/2_diff\n",layer_id);
		}



	layer_list[layer_id].train_inf.para1_diff=cuda_make_float_array(ip_w_size);
	layer_list[layer_id].train_inf.para1_diff_hist=cuda_make_float_array(ip_w_size);
	layer_list[layer_id].train_inf.para1_size=ip_w_size;
	if(layer_list[layer_id].train_inf.en_bias){
		layer_list[layer_id].train_inf.para2_diff=cuda_make_float_array(layer_list[layer_id].output_num);
		layer_list[layer_id].train_inf.para2_diff_hist=cuda_make_float_array(layer_list[layer_id].output_num);
		layer_list[layer_id].train_inf.para2_size=layer_list[layer_id].output_num;
		}

	cuda_set(ip_w_size,0, layer_list[layer_id].train_inf.para1_diff);
	cuda_set(ip_w_size,0, layer_list[layer_id].train_inf.para1_diff_hist);
	if(layer_list[layer_id].train_inf.en_bias){
		cuda_set(layer_list[layer_id].output_num,0, layer_list[layer_id].train_inf.para2_diff);
		cuda_set(layer_list[layer_id].output_num,0, layer_list[layer_id].train_inf.para2_diff_hist);
		}

	return 0;
}



int 	net_ip_forward(t_layer_info*  layer_list,int layer_id, float* input, int ch, int h, int w,float* output)
{
	int o_h;
	int o_w;
	int isize=ch*h*w;
	int out_ch = layer_list[layer_id].output_num;

	inner_product_without_sparse(layer_list,layer_id,input, isize,  output);

	//print_gpu_data(output , out_ch);
	return 1;
}



int net_ip_forward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz)
{
	int i_c=layer_list[layer_id].in_dim[0][0];
	int i_h=layer_list[layer_id].in_dim[0][1];
	int i_w=layer_list[layer_id].in_dim[0][2];

	int bottom_size=layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	for(int i=0;i<batch_sz;i++){
		float* bottom_=layer_list[layer_id].input_buf[0]+i*3*bottom_size;			
		float* top_=layer_list[layer_id].output_buf+i*3*top_size;
		net_ip_forward( layer_list, layer_id, bottom_, i_c, i_h, i_w,  top_);
		}
}



int 	net_ip_backward(t_layer_info*  layer_list,int layer_id,float* bottom, float* top)
{
	//step1, claculate the ip weight differ
	int bottom_size=layer_list[layer_id].in_dim[0][2]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][0];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	float* bottom_data=bottom;
	float* bottom_diff=bottom+bottom_size;
	float* top_diff=top+top_size;
	int ip_w_size=layer_list[layer_id].output_num*layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];

	float* bottom_diff_temp=bottom+bottom_size*2;

	caffe_gpu_gemm(CblasNoTrans,
		    CblasNoTrans, top_size, bottom_size, 1,
		    1, top_diff, bottom_data, 1,
		    layer_list[layer_id].train_inf.para1_diff);

	if(layer_list[layer_id].train_inf.en_bias){
		float* multiplier_=cuda_make_float_array(1);
		cuda_set(1,1.0,multiplier_);
		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, top_size, 1, 1, 1., top_diff, multiplier_,	1., layer_list[layer_id].train_inf.para2_diff);
		cuda_free(multiplier_);
		}

	caffe_gpu_gemm(CblasNoTrans, CblasNoTrans,
	          1, bottom_size, top_size,
	          1, top_diff, layer_list[layer_id].para1_entry,
	          0, bottom_diff_temp);

	caffe_gpu_axpy(bottom_size, 1.0, bottom_diff_temp,bottom_diff);

#if 0
	print_gpu_data(bottom_diff, bottom_size);
	print_stop_sign("ip back \n");
#endif

	if(layer_list[layer_id].train_inf.para_diff_way==e_para1_prune){
		float* para1_data_host=new float[ip_w_size];
		cuda_pull_array(layer_list[layer_id].train_inf.para1_diff, para1_data_host, ip_w_size);
		caffe_mul(ip_w_size, para1_data_host, layer_list[layer_id].train_inf.para_diff_mask, para1_data_host);
		cuda_push_array(layer_list[layer_id].train_inf.para1_diff, para1_data_host, ip_w_size);
		delete []para1_data_host;
	}else if(layer_list[layer_id].train_inf.para_diff_way==e_para_unchange){
		cuda_set(ip_w_size,0,layer_list[layer_id].train_inf.para1_diff);
		if(layer_list[layer_id].train_inf.en_bias){
			cuda_set(layer_list[layer_id].output_num,0,layer_list[layer_id].train_inf.para2_diff);
			}
		}
	return 1;
}


int net_ip_backward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz)
{
	int bottom_size=layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	for(int i=0;i<batch_sz;i++){
		float* bottom_=layer_list[layer_id].input_buf[0]+i*3*bottom_size;			
		float* top_=layer_list[layer_id].output_buf+i*3*top_size;
		net_ip_backward( layer_list,layer_id, bottom_, top_);
		}
}

