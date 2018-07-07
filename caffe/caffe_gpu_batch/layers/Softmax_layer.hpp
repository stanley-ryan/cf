#ifndef	_SOFTMAX_LAYER_H
#define 	_SOFTMAX_LAYER_H

#include<stdio.h>
#include<stdlib.h>
#include"../netsolve.hpp"

extern	int 	net_softmax_forward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz);
extern	int 	net_softmax_initial_para(t_layer_info*  layer_list,int layer_id,int in_ch);


#endif

