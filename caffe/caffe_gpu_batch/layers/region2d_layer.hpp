#ifndef	_REGION2D_LAYER_H
#define 	_REGION2D_LAYER_H

#include<stdio.h>
#include<stdlib.h>
#include"../netsolve.hpp"

extern	int 	net_region2d_loss_forward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz,t_box_list*  true_box_list,string im_file[]);
extern	int 	net_region2d_loss_initial_para(t_layer_info*  layer_list,int layer_id,int in_ch);
extern	int 	net_region2d_loss_backward_batch(t_layer_info*  layer_list,int layer_id, int batch_sz, t_box_list* true_box_list);



#define 	CLASS_NUM 						80
#define 	REGION2D_ANCHOR_BOX_NUM		5

#define 	MAX_TRUE_BOX_PER_IMG			30


typedef struct box_info{
	t_rgbox 	box;
	float 	obj_conf;
	float	 	class_prob[CLASS_NUM];
}t_boxinf;

#define BOX_STRIDE	(5+CLASS_NUM)

#endif

