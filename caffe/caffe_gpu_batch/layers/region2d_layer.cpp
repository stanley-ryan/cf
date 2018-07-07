
#include <iostream>
#include <string.h>
#include<float.h>
#include<cmath>
#include<math.h>
#include"gemm.hpp"
#include"region2d_layer.hpp"
#include"cuda.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;



void   show_object(string im_file, t_box_list &box_list)
{
	Mat im = imread(im_file);
	int im_h=im.rows;
	int im_w=im.cols;
	for(int i=0;i<box_list.size();i++){
		int x0=box_list[i].first.left*im_w;
		int y0=box_list[i].first.top*im_h;
		int x1=box_list[i].first.right*im_w;
		int y1=box_list[i].first.bottom*im_h;
		rectangle(im,cvPoint(x0,y0),cvPoint(x1,y1),cvScalar(255,255,255),5);
		}
	imshow("show_img2",im);
	waitKey(1);
}


int 	net_region2d_loss_initial_para(t_layer_info*  layer_list,int layer_id,int in_ch)
{
	layer_list[layer_id].train_inf.para1_size=REGION2D_ANCHOR_BOX_NUM*2;		//9 anchor box
	layer_list[layer_id].train_inf.para2_size=5;		//lamda for loss(object,no-object, class, coord,iou_thresh)
	layer_list[layer_id].train_inf.para2_bp_mode=e_para_unbp;
	layer_list[layer_id].train_inf.para1_bp_mode=e_para_unbp;

	return 0;
}


t_rgbox convert2float_box(t_rgbox box_in, int cell_rows, int cell_cols, int cell_r, int cell_c, int b,float *anchbox_bias)
{
	t_rgbox box_ret;
	box_ret.x=box_in.x*cell_cols-cell_c;
	box_ret.y=box_in.y*cell_rows-cell_r;
	box_ret.w=log(box_in.w*cell_cols / anchbox_bias[2*b]);
	box_ret.h=log(box_in.h*cell_rows / anchbox_bias[2*b + 1]);

	return box_ret;
}


t_rgbox get_float_box(float *bottom_data, int cell_rows, int cell_cols, int cell_r, int cell_c, int b)
{
	t_rgbox box;
	int cell_dim=cell_rows*cell_cols;
	int boxcell_dim=cell_dim*BOX_STRIDE;
	int entry_idx=b*boxcell_dim+cell_r*cell_cols+cell_c;
	box.x=cell_c+bottom_data[entry_idx];
	box.y=cell_r+bottom_data[entry_idx+cell_dim];
	box.w=bottom_data[entry_idx+cell_dim*2];
	box.h=bottom_data[entry_idx+cell_dim*3];
	return box;
}

void update_bottom_diff(float *bottom_diff, int cell_rows, int cell_cols,float* conf_delta, float* box_delta, float* class_delta)
{
	int cell_dim=cell_rows*cell_cols;
	int boxcell_dim=cell_dim*BOX_STRIDE;
	for(int b=0;b<REGION2D_ANCHOR_BOX_NUM;b++){
		for(int r=0;r<cell_rows;r++){
			for(int c=0;c<cell_cols;c++){
				int box_idx=b*cell_rows*cell_cols+r*cell_cols+c;
				int entry_idx=b*boxcell_dim+r*cell_cols+c;
				bottom_diff[entry_idx]+=box_delta[4*box_idx];
				bottom_diff[entry_idx+cell_dim]+=box_delta[4*box_idx+1];
				bottom_diff[entry_idx+cell_dim*2]+=box_delta[4*box_idx+2];
				bottom_diff[entry_idx+cell_dim*3]+=box_delta[4*box_idx+3];
				bottom_diff[entry_idx+cell_dim*4]+=conf_delta[box_idx];
				for(int i=0;i<CLASS_NUM;i++){
					bottom_diff[entry_idx+cell_dim*(5+i)]+=class_delta[CLASS_NUM*box_idx+i];
					}
				}
			}
		}
}


t_boxinf get_exact_boxinfo(float *bottom_data, int cell_rows, int cell_cols, int cell_r, int cell_c, int b,float *anchbox_bias)
{
	t_boxinf box;
	int cell_dim=cell_rows*cell_cols;
	int boxcell_dim=cell_dim*BOX_STRIDE;
	int entry_idx=b*boxcell_dim+cell_r*cell_cols+cell_c;
	box.box.x=(cell_c+bottom_data[entry_idx])/cell_cols;
	box.box.y=(cell_r+bottom_data[entry_idx+cell_dim])/cell_rows;
	box.box.w=exp(bottom_data[entry_idx+cell_dim*2])* anchbox_bias[2*b] /cell_cols;
	box.box.h=exp(bottom_data[entry_idx+cell_dim*3])* anchbox_bias[2*b+1] /cell_rows;
	box.obj_conf=bottom_data[entry_idx+cell_dim*4];
	for(int i=0;i<CLASS_NUM;i++){
		box.class_prob[i]=bottom_data[entry_idx+cell_dim*(5+i)];
		}
    	return box;
}



float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}


float box_intersection(t_rgbox a, t_rgbox b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(t_rgbox a, t_rgbox b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}


float box_iou(t_rgbox a, t_rgbox b)
{
    return box_intersection(a, b)/box_union(a, b);
}


float get_best_iou_from_truebox_list(t_rgbox pred_box, t_box_list &true_box_label)
{
	float best_iou=0;
	for(int n=0;n<true_box_label.size();n++){
		t_rgbox true_box=true_box_label[n].first;
		if(true_box.x<=0){
			break;
			}
		float iou_rate=box_iou(pred_box,true_box);
		if(best_iou<iou_rate){
			best_iou=iou_rate;
			}
		}
	return best_iou;
}

float cal_region_loss(float* conf_delta, float* box_delta, float* class_delta, int total_anchorbox)
{
	float loss=0;
	for(int i=0;i<total_anchorbox;i++){
		loss+=conf_delta[i]*conf_delta[i];
		//printf("%f,",conf_delta[i]);
		}
	//printf("conf_delta  ------\n");
	//getchar();
	for(int i=0;i<total_anchorbox*4;i++){
		loss+=box_delta[i]*box_delta[i];
		//printf("%f,",box_delta[i]);
		}
	//printf("box_delta  ------\n");
	//getchar();

	for(int i=0;i<total_anchorbox*CLASS_NUM;i++){
		loss+=class_delta[i]*class_delta[i];
		//printf("%f,",class_delta[i]);
		}
	//printf("class_delta  ------\n");
	//getchar();

	loss=std::sqrt(loss);
	//printf("#################################loss=%f\n",loss);
	//getchar();

	return loss;
}



int 	net_region2d_loss_forward(t_layer_info*  layer_list,int layer_id, float* bottom, int ch, int h, int w,float* top, t_box_list true_box_list,string im_file)
{
	static int forward_cnt=0;
	float 	loss=0;
	//t_box_list true_box_label=truth[0].first;
	//int class_tag=truth[0].second;
	int cell_dim=layer_list[layer_id].in_dim[0][0];
	int cell_rows=layer_list[layer_id].in_dim[0][1];
	int cell_cols=layer_list[layer_id].in_dim[0][2];
	int 		bottom_size=layer_list[layer_id].in_dim[0][2]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][0];
	float* 	bottom_data=bottom;
	float* bottom_diff=bottom+bottom_size;
	float* bottom_diff_temp=bottom+bottom_size*2;

	float* 	top_data=top;

	float		lboj=layer_list[layer_id].para2_entry[0];
	float		lnoboj=layer_list[layer_id].para2_entry[1];
	float 	lclass=layer_list[layer_id].para2_entry[2];
	float 	lcoord=layer_list[layer_id].para2_entry[3];
	float 	Iou_thresh=layer_list[layer_id].para2_entry[4];
	float*	anchbox_bias=layer_list[layer_id].para1_entry;

	float* cell_data_host=new float[bottom_size];
	cuda_pull_array(bottom_data, cell_data_host, bottom_size);

	int total_anchorbox=REGION2D_ANCHOR_BOX_NUM*cell_rows*cell_cols;
	float*  conf_delta=new float[total_anchorbox];
	float*  box_delta=new float[total_anchorbox*4];
	float*  class_delta=new float[total_anchorbox*CLASS_NUM];
	top_data[0]=0;
	top_data[1]=0;
	top_data[2]=0;
	top_data[3]=0;
    	memset(conf_delta, 0, total_anchorbox*sizeof(float));
    	memset(box_delta, 0, total_anchorbox*4*sizeof(float));
    	memset(class_delta, 0, total_anchorbox*CLASS_NUM*sizeof(float));

	//show_object(im_file, true_box_list);

	//calculate no-object confidence_delta
	for(int b=0;b<REGION2D_ANCHOR_BOX_NUM;b++){
		for(int r=0;r<cell_rows;r++){
			for(int c=0;c<cell_cols;c++){
				int box_idx=b*cell_rows*cell_cols+r*cell_cols+c;
				t_boxinf pred_box= get_exact_boxinfo(cell_data_host, cell_rows, cell_cols, r, c, b, anchbox_bias);
				float best_iou=get_best_iou_from_truebox_list(pred_box.box, true_box_list);
				if(best_iou>Iou_thresh){		//noobject confidence
					conf_delta[box_idx]=0;
				}else{						//noobj confidence
					conf_delta[box_idx]=lnoboj*(pred_box.obj_conf-0);	//set true 0, this cell definitly to haven't the object
					}
				//TBD, if farward<12800, delta_rangebox()
				if(forward_cnt++<12800){
					t_rgbox float_anchorbox;
					float_anchorbox.x = (c + .5)/cell_cols;
					float_anchorbox.y = (r + .5)/cell_rows;
					float_anchorbox.w = anchbox_bias[2*b]/cell_cols;
					float_anchorbox.h = anchbox_bias[2*b+1]/cell_rows;
					int box_idx=b*cell_rows*cell_cols+r*cell_cols+c;

					//t_rgbox anchorbox =convert2float_box(exact_anchorbox,  cell_rows,  cell_cols, r, c,  b,anchbox_bias);

					box_delta[box_idx*4]    = 0.1*(pred_box.box.x-float_anchorbox.x);		//diff(x)
					box_delta[box_idx*4+1]=0.1*(pred_box.box.y-float_anchorbox.y);		//diff(y)
					box_delta[box_idx*4+2]=0.1*(pred_box.box.w-float_anchorbox.w);		//diff(log(w))
					box_delta[box_idx*4+3]=0.1*(pred_box.box.h-float_anchorbox.h);		//diff(log(w))
					}
				}
			}
		}

	//calculate obj-confidence_delta,obj-box_delta,obj_class_delta

	for(int n=0;n<true_box_list.size();n++){
		t_rgbox true_box=true_box_list[n].first;
		int class_tag=true_box_list[n].second;
		if(true_box.x<=0){
			break;
			}
		top_data[1]+=1;
		int c=true_box.x*cell_cols;
		int r=true_box.y*cell_rows;
		int best_b=0;
		float best_iou=0;
		t_boxinf best_pred_box;
		for(int b=0;b<REGION2D_ANCHOR_BOX_NUM;b++){
			t_boxinf pred_box= get_exact_boxinfo(cell_data_host, cell_rows, cell_cols, r, c, b, anchbox_bias);
			float iou_rate=box_iou(true_box,pred_box.box);
			if(iou_rate>best_iou){
				best_iou=iou_rate;
				best_b=b;
				best_pred_box=pred_box;
				}
			}
		if(best_iou>Iou_thresh){
			top_data[2]+=1;
			}
		int best_box_idx=best_b*cell_rows*cell_cols+r*cell_cols+c;
		t_rgbox best_pred_box_float=get_float_box(cell_data_host, cell_rows, cell_cols, r, c, best_b);
		t_rgbox true_box_float =convert2float_box(true_box,  cell_rows,  cell_cols, r, c,  best_b,anchbox_bias);

		conf_delta[best_box_idx]=lboj*(best_pred_box.obj_conf-1);
		box_delta[best_box_idx*4]=	lcoord*(best_pred_box_float.x-true_box_float.x);		//diff(x)
		box_delta[best_box_idx*4+1]=lcoord*(best_pred_box_float.y-true_box_float.y);		//diff(y)
		box_delta[best_box_idx*4+2]=lcoord*(best_pred_box_float.w-true_box_float.w);		//diff(log(w))
		box_delta[best_box_idx*4+3]=lcoord*(best_pred_box_float.h-true_box_float.h);		//diff(log(w))

		for(int i=0;i<CLASS_NUM;i++){
			if(class_tag==i){
				class_delta[best_box_idx*CLASS_NUM+i]=lclass*(best_pred_box.class_prob[i]-1);
				if(best_iou>Iou_thresh){
					top_data[3]+=1;
					}
			}else{
				class_delta[best_box_idx*CLASS_NUM+i]=lclass*(best_pred_box.class_prob[i]-0);
				}
			}
		}


	loss= cal_region_loss(conf_delta, box_delta, class_delta, total_anchorbox);
	top_data[0]=loss;

	//apply the delta to bottom_diff
	float* bottom_diff_host=new float[bottom_size];
	cuda_pull_array(bottom_diff, bottom_diff_host, bottom_size);
	update_bottom_diff(bottom_diff_host,  cell_rows,  cell_cols, conf_delta, box_delta, class_delta);
	cuda_push_array(bottom_diff, bottom_diff_host, bottom_size);


	delete []bottom_diff_host;
	delete []conf_delta;
	delete []box_delta;
	delete []class_delta;
	delete []cell_data_host;
	printf("region loss layer ===>[%f,%f,%f,%f]\n",top_data[0],top_data[1],top_data[2],top_data[3]);

	return 1;
}



int net_region2d_loss_forward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz,t_box_list*  true_box_list,string im_file[])
{
	int i_c=layer_list[layer_id].in_dim[0][0];
	int i_h=layer_list[layer_id].in_dim[0][1];
	int i_w=layer_list[layer_id].in_dim[0][2];

	int bottom_size=layer_list[layer_id].in_dim[0][0]*layer_list[layer_id].in_dim[0][1]*layer_list[layer_id].in_dim[0][2];
	int top_size=layer_list[layer_id].out_dim[0]*layer_list[layer_id].out_dim[1]*layer_list[layer_id].out_dim[2];
	for(int i=0;i<batch_sz;i++){
		float* bottom_=layer_list[layer_id].input_buf[0]+i*3*bottom_size;
		float* top_=layer_list[layer_id].output_buf+i*3*top_size;
		net_region2d_loss_forward( layer_list, layer_id, bottom_, i_c, i_h, i_w,  top_,true_box_list[i], im_file[i]);
		}
}



int 	net_region2d_loss_backward(t_layer_info*  layer_list,int layer_id, t_box_list true_box_list)
{
	return 1;
}

int 	net_region2d_loss_backward_batch(t_layer_info*  layer_list,int layer_id, int batch_sz, t_box_list* true_box_list)
{
	return 1;
}

