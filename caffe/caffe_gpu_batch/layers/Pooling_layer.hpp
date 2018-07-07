#ifndef	_POOL_LAYER_H
#define 	_POOL_LAYER_H

#include<stdio.h>
#include<stdlib.h>
#include"../netsolve.hpp"
#include<float.h>

extern	int 	net_pool_forward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz);
extern	int 	net_pool_initial_para(t_layer_info*  layer_list,int layer_id,int in_ch);
extern	int 	net_pool_backward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz);


#endif

