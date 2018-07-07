#include"Center_loss_layer.hpp"
#include <iostream>
#include <string.h>
#include<float.h>
#include<cmath>
#include<math.h>
#include"gemm.hpp"


int 	net_center_loss_initial_para(t_layer_info*  layer_list,int layer_id,int in_ch)
{
	int center_dim=layer_list[layer_id].in_dim[0][2]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][0];
	int class_num=layer_list[layer_id].output_num;

	layer_list[layer_id].train_inf.para1_size=center_dim*class_num;
	layer_list[layer_id].train_inf.para1_diff=new float[layer_list[layer_id].train_inf.para1_size];
	layer_list[layer_id].train_inf.para1_diff_hist=new float[layer_list[layer_id].train_inf.para1_size];
	layer_list[layer_id].train_inf.para1_bp_mode=e_para_unbp;
	layer_list[layer_id].train_inf.para2_size=0;


	int para_size;
	int fan_in=center_dim;

	para_size=center_dim*class_num;
	if(layer_list[layer_id].train_inf.para1_ini_way==para_ini_gaussian){
		gaussian_fill(0.01,0.0,layer_list[layer_id].para1_entry, para_size);
		printf("initial para1 centerloss_layer_%d: initial parameter gaussian\n",layer_id);
	}else if(layer_list[layer_id].train_inf.para1_ini_way==para_ini_xavier){
		float scale=sqrt(3.0/fan_in);
		unirom_fill(-scale, scale, layer_list[layer_id].para1_entry, para_size);
		printf("initial para1 centerloss_layer_%d: initial parameter uninorm\n",layer_id);
	}else if(layer_list[layer_id].train_inf.para1_ini_way==para_ini_loading){
		printf("initial para1 centerloss_layer_%d: loading parameter\n",layer_id);
	}else{
		printf("initial para1 centerloss_layer_%d: invalid parameter initial way\n",layer_id);
		exit(-1);
		}

	caffe_set(center_dim*class_num,0, layer_list[layer_id].train_inf.para1_diff);
	caffe_set(center_dim*class_num,0, layer_list[layer_id].train_inf.para1_diff_hist);

	return 0;
}





int 	net_center_loss_forward(t_layer_info*  layer_list,int layer_id, float* bottom, int ch, int h, int w,float* top, t_label label)
{
	float loss=0;
	int bottom_size=layer_list[layer_id].in_dim[0][2]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][0];
	int center_dim=bottom_size;
	int class_num=layer_list[layer_id].output_num;

	float* center=layer_list[layer_id].para1_entry;
	float* bottom_data=bottom;
	float* bottom_diff=bottom+bottom_size;
	float* top_data=top;

	//for(int i=0;i<layer_list[layer_id].train_inf.para1_size; i++){
	//	printf("%f,", center[i]);
	//	}
	//printf("center update \n");
	//getchar();


	if(label!=EMPTY_LABEL){
		int ll=label;
		for(int j=0;j<bottom_size;j++){
			loss+=(bottom_data[j]-center[ll*center_dim+j])*(bottom_data[j]-center[ll*center_dim+j]);
			}
		loss=loss/2;
		top_data[0]=loss;
	}

	float centerdistance=99999999;
	int  	near_item=-1;
	float dist=0.0;
	for(int ll=0;ll<class_num;ll++){
		for(int j=0;j<bottom_size;j++){
			dist+=(bottom_data[j]-center[ll*center_dim+j])*(bottom_data[j]-center[ll*center_dim+j]);
			}
		dist=dist/2;
		if(dist<centerdistance){
			centerdistance=dist;
			near_item=ll;
			}
		}
	top_data[1]=near_item;
	top_data[2]=centerdistance;


	if(label!=EMPTY_LABEL){
		printf("center_loss=%f, label=%d\n",loss,label);
	}else{
		printf("predict item=%d, dist=%f\n",near_item,centerdistance);
		}
	//getchar();
	return 1;
}


int net_center_loss_forward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz,t_label*  label_list)
{
	int i_c=layer_list[layer_id].in_dim[0][0];
	int i_h=layer_list[layer_id].in_dim[0][1];
	int i_w=layer_list[layer_id].in_dim[0][2];

	int bottom_size=layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	for(int i=0;i<batch_sz;i++){
		float* bottom_=layer_list[layer_id].input_buf[0]+i*3*bottom_size;			
		float* top_=layer_list[layer_id].output_buf+i*3*top_size;
		net_center_loss_forward( layer_list, layer_id, bottom_, i_c, i_h, i_w,  top_,label_list[i]);
		}
}



int 	net_center_loss_backward(t_layer_info*  layer_list,int layer_id, t_label label,float* bottom, float* top)
{
	float loss=0;
	int bottom_size=layer_list[layer_id].in_dim[0][2]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][0];
	int center_dim=bottom_size;
	float* center=layer_list[layer_id].para1_entry;
	float* bottom_data=bottom;
	float* bottom_diff=bottom+bottom_size;
	float* top_data=top;
	float* center_diff=layer_list[layer_id].train_inf.para1_diff;
	float center_loss_weight=layer_list[layer_id].para2_entry[0];
	float* bottom_diff_temp=bottom+bottom_size*2;


	int ll=label;

	for(int j=0;j<bottom_size;j++){
		bottom_diff_temp[j]=bottom_data[j]-center[ll*center_dim+j];
		center_diff[ll*center_dim+j]+=0.5*(center[ll*center_dim+j] -bottom_data[j]);
		}

	printf("centerloss_weight=%f\n",center_loss_weight);
	caffe_scal(bottom_size, center_loss_weight, bottom_diff_temp);

#if 0
	for(int i=0;i<bottom_size;i++){
		printf("%f,",bottom_diff_temp[i]);
		}
	printf("centerloss %d bottom diff backward_size: [ %d]\n",layer_id,   bottom_size);
	getchar();
#endif

	caffe_axpy(bottom_size, 1.0, bottom_diff_temp,bottom_diff);

	return 1;
}


int net_center_loss_backward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz,t_label*  label_list)
{
	int bottom_size=layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	for(int i=0;i<batch_sz;i++){
		float* bottom_=layer_list[layer_id].input_buf[0]+i*3*bottom_size;			
		float* top_=layer_list[layer_id].output_buf+i*3*top_size;
		net_center_loss_backward( layer_list, layer_id, label_list[i], bottom_, top_);
		}
}

