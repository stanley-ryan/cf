#include "LRN_layer.hpp"
#include <iostream>
#include <string.h>
#include<pthread.h>


int acc_sqr_across_channel(float* input, int ic, int ih, int iw,float* sqr_acc)
{

	int c,h,w;
	int dim_area=ih*iw;
	for(h=0;h<ih;h++){
		for(w=0;w<iw;w++){
			sqr_acc[h*iw+w]=input[h*iw+w]*input[h*iw+w];
			}
		}

	for(h=0;h<ih;h++){
		for(w=0;w<iw;w++){
			for(c=1;c<ic;c++){
				sqr_acc[h*iw+w+c*dim_area]=input[h*iw+w+c*dim_area]*input[h*iw+w+c*dim_area]+sqr_acc[h*iw+w+(c-1)*dim_area];
				}
			}
		}
	return dim_area;
}

int cal_lrn_across_channel(float* input,float* sqr_acc, float alpha, float beta, int local_size, int ic, int ih, int iw,float* output)
{

	int c,h,w;
	int dim_area=ih*iw;
	int half_local_size=(local_size-1)/2;
	float alpha_n=alpha/local_size;
	float acc_start,acc_end;

	for(h=0;h<ih;h++){
		for(w=0;w<iw;w++){
			for(c=0;c<ic;c++){
				if((c-half_local_size-1)<0){
					acc_start=(float)0;
				}else{
					acc_start=sqr_acc[h*iw+w+(c-half_local_size-1)*dim_area];
				}
				if((c+half_local_size)>=ic){
					acc_end=sqr_acc[h*iw+w+(ic-1)*dim_area];
				}else{
					acc_end=sqr_acc[h*iw+w+(c+half_local_size)*dim_area];
				}
				output[h*iw+w+c*dim_area]=input[h*iw+w+c*dim_area]/pow((1+alpha_n*(acc_end-acc_start)),beta);
				}
			}
		}

	return dim_area;
}


int 	net_lrn_initial_para(t_layer_info*  layer_list,int layer_id,int in_ch)
{
	//layer_list[layer_id].train_inf.para1_size=0;
	//layer_list[layer_id].train_inf.para2_size=0;

	return 0;
}



int 	net_lrn_forward(t_layer_info*  layer_list,int layer_id, float* bottom, int ch, int h, int w,float* top)
{
	int i;
	int dim=h*w;
	int channels=ch;
	int count=dim*ch;
	float* alpha=layer_list[layer_id].para1_entry;
	float* beta=layer_list[layer_id].para2_entry;
	float* sqr_acc=new float[count];
	int local_size=layer_list[layer_id].ker_nfo.ker_size;
	if(local_size%2==0){
		printf("the local size shall be odder value\n");
		exit(-1);
		}

	switch(layer_list[layer_id].ker_nfo.act){
		case lrn_across_chan:
			acc_sqr_across_channel(bottom, ch, h,w,sqr_acc);
			cal_lrn_across_channel(bottom,sqr_acc, alpha[0], beta[0], local_size, ch, h, w,top);
			break;
		case lrn_within_chan:
			//TBD
			break;

		}
	delete []sqr_acc;
	return 1;
}


int net_lrn_forward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz)
{
	int i_c=layer_list[layer_id].in_dim[0][0];
	int i_h=layer_list[layer_id].in_dim[0][1];
	int i_w=layer_list[layer_id].in_dim[0][2];

	int bottom_size=layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	for(int i=0;i<batch_sz;i++){
		float* bottom_=layer_list[layer_id].input_buf[0]+i*3*bottom_size;			
		float* top_=layer_list[layer_id].output_buf+i*3*top_size;
		net_lrn_forward( layer_list, layer_id, bottom_, i_c, i_h, i_w,  top_);
		}
}

