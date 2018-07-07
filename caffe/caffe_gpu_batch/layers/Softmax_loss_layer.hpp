#ifndef	_SOFTMAX_LOSS_LAYER_H
#define 	_SOFTMAX_LOSS_LAYER_H

#include<stdio.h>
#include<stdlib.h>
#include"../netsolve.hpp"

extern	int 	net_softmax_loss_forward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz,t_label* label_list);
extern	int 	net_softmax_loss_initial_para(t_layer_info*  layer_list,int layer_id,int in_ch);
extern	int 	net_softmax_loss_backward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz,t_label* label_list);


#endif

