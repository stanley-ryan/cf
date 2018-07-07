#include"logistic_loss_layer_sf.hpp"
#include <iostream>
#include <string.h>
#include<float.h>
#include<cmath>
#include<math.h>

#include"gemm.hpp"
#include"cuda.hpp"


int 	net_logistic_sf_loss_initial_para(t_layer_info*  layer_list,int layer_id,int in_ch)
{
	int alpha_size=layer_list[layer_id].in_dim[0][2]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][0];

	float* logreg_w=new float[alpha_size+1];
	if(layer_list[layer_id].train_inf.para1_ini_way!=para_ini_loading){
		printf("initial para1 logreg_loss %d: initial parameter by setting as %f \n",layer_id,(float)1/alpha_size);
		for(int i=0;i<alpha_size+1;i++){
			logreg_w[i]=(float)1/alpha_size;
			}
		layer_list[layer_id].para1_entry=cuda_make_array(logreg_w,alpha_size+1);
	}else{
		printf("initial para1 logreg_loss %d: loading parameter\n",layer_id);
		layer_list[layer_id].para1_entry=cuda_make_array(layer_list[layer_id].para1_entry,alpha_size+1);
		}

	if(layer_list[layer_id].train_inf.para_diff_way==e_para_unchange){
		printf("layer id =%d, all zero for para1/2_diff\n",layer_id);
		}

	layer_list[layer_id].train_inf.para1_diff=cuda_make_float_array(alpha_size+1);
	layer_list[layer_id].train_inf.para1_diff_hist=cuda_make_float_array(alpha_size+1);
	layer_list[layer_id].train_inf.para1_size=alpha_size+1;
	layer_list[layer_id].train_inf.para2_size=1;
	layer_list[layer_id].train_inf.para2_bp_mode=e_para_unbp;
	cuda_set(alpha_size+1,0, layer_list[layer_id].train_inf.para1_diff);
	cuda_set(alpha_size+1,0, layer_list[layer_id].train_inf.para1_diff_hist);

	delete []logreg_w;
	return 0;
}



static float net_logistic_sf_loss_cal_score(int y,float* weights, float* x, int feat_dim)		//y=> {1 ,-1}
{
	int i;
	float score=0;
	for(i=0;i<feat_dim;i++){
		score=score+weights[i]*x[i];
		}
	score=score+weights[feat_dim]*1;		//add the k bias, xk=1
	score=y*score;
	return score;
}



static float net_logistic_sf_cal_prob(float score)
{
	float prob;
	prob=1/(1+exp(-score));
	return prob;
}


static float net_logistic_sf_cal_loss(float score)
{
	float loss;
	loss=log(1+exp(-score));
	return loss;
}


int 	net_logistic_sf_loss_forward(t_layer_info*  layer_list,int layer_id, float* bottom, int ch, int h, int w,float* top, t_label label)
{
	float 	loss=0;
	int 		bottom_size=layer_list[layer_id].in_dim[0][2]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][0];
	float* 	bottom_data=bottom;
	float* 	top_data=top;
	float* 	weights=layer_list[layer_id].para1_entry;
	float 	logistic_loss_weight=layer_list[layer_id].para2_entry[0];
	int 		alpha_size=layer_list[layer_id].in_dim[0][2]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][0];

	float 	prob_value;

	float* bottom_data_host=new float[bottom_size];
	float* weights_data_host=new float[alpha_size+1];

	cuda_pull_array(bottom_data, bottom_data_host, bottom_size);
	cuda_pull_array(weights, weights_data_host, alpha_size+1);

	if(label!=EMPTY_LABEL){
		prob_value=net_logistic_sf_cal_prob(net_logistic_sf_loss_cal_score(label, weights_data_host, bottom_data_host, bottom_size));
		loss=net_logistic_sf_cal_loss(net_logistic_sf_loss_cal_score(label, weights_data_host, bottom_data_host, bottom_size));
		top_data[0]=loss;		
	}

	float log_value=net_logistic_sf_cal_prob(net_logistic_sf_loss_cal_score(1.0, weights_data_host, bottom_data_host, bottom_size));

	if(log_value>0.5){
		top_data[1]=1;
	}else{
		top_data[1]=-1;
		}
	top_data[2]=prob_value;

#if 0
	if(label!=EMPTY_LABEL){
		printf("logistic layer_loss=%f,prob=%f,\n",top_data[0],prob_value);
	}else{
		printf("loss=%f, label=%d, prob=%f,logistic_loss_weight=%f\n",top_data[0], top_data[1], top_data[2],logistic_loss_weight);
		}
	//getchar();
#endif
	delete []bottom_data_host;
	delete []weights_data_host;

	return 1;
}


int net_logistic_sf_loss_forward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz,t_label*  label_list)
{
	int i_c=layer_list[layer_id].in_dim[0][0];
	int i_h=layer_list[layer_id].in_dim[0][1];
	int i_w=layer_list[layer_id].in_dim[0][2];

	int bottom_size=layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	for(int i=0;i<batch_sz;i++){
		float* bottom_=layer_list[layer_id].input_buf[0]+i*3*bottom_size;			
		float* top_=layer_list[layer_id].output_buf+i*3*top_size;
		net_logistic_sf_loss_forward( layer_list, layer_id, bottom_, i_c, i_h, i_w,  top_,label_list[i]);
		}
}




int 	net_logistic_sf_loss_backward(t_layer_info*  layer_list,int layer_id, t_label label,float* bottom, float* top)
{
	float loss=0;
	int bottom_size=layer_list[layer_id].in_dim[0][2]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][0];
	float* bottom_data=bottom;
	float* bottom_diff=bottom+bottom_size;
	float* bottom_diff_temp=bottom+bottom_size*2;

	float* top_data=top;
	float logistic_loss_weight=layer_list[layer_id].para2_entry[0];
	float  prob_value=top_data[2];
	float* weights=layer_list[layer_id].para1_entry;
	float* weights_diff=layer_list[layer_id].train_inf.para1_diff;
	int 	alpha_size=layer_list[layer_id].in_dim[0][2]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][0];

	float* bottom_diff_host=new float[bottom_size];
	float* bottom_data_host=new float[bottom_size];
	float* weights_data_host=new float[alpha_size+1];
	float* weightsdiff_data_host=new float[alpha_size+1];

	cuda_pull_array(bottom_diff, bottom_diff_host, bottom_size);
	cuda_pull_array(bottom_data, bottom_data_host, bottom_size);
	cuda_pull_array(weights, weights_data_host, alpha_size+1);
	cuda_pull_array(weights_diff, weightsdiff_data_host, alpha_size+1);

	for(int i=0;i<bottom_size;i++){
		bottom_diff_host[i]+=(1-prob_value)*(-label*weights_data_host[i]);
		weightsdiff_data_host[i]+=(1-prob_value)*(-label*bottom_data_host[i]);
		}
	weightsdiff_data_host[bottom_size]+=(1-prob_value)*(-label*1);

	cuda_push_array(bottom_diff, bottom_diff_host, bottom_size);

	if(layer_list[layer_id].train_inf.para_diff_way==e_para_unchange){
		cuda_set(bottom_size+1,0,weights_diff);
	}else{
		cuda_push_array(weights_diff, weightsdiff_data_host, alpha_size+1);
		}

	delete []bottom_diff_host;
	delete []bottom_data_host;
	delete []weights_data_host;
	delete []weightsdiff_data_host;

	return 1;
}



int net_logistic_sf_loss_backward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz,t_label*  label_list)
{
	int bottom_size=layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	for(int i=0;i<batch_sz;i++){
		float* bottom_=layer_list[layer_id].input_buf[0]+i*3*bottom_size;			
		float* top_=layer_list[layer_id].output_buf+i*3*top_size;
		net_logistic_sf_loss_backward( layer_list, layer_id, label_list[i], bottom_, top_);
		}
}

