#ifndef	_CONCAT_LAYER_H
#define _CONCAT_LAYER_H

#include<stdio.h>
#include<stdlib.h>
#include"../netsolve.hpp"

extern	int 	net_Concat_forward(t_layer_info*  layer_list,int layer_id, float* input1,float* input2 ,float* input3,float* input4 ,float* output);
extern	int 	net_concat_initial_para(t_layer_info*  layer_list,int layer_id,int in_ch);
extern	int 	net_concat_backward(t_layer_info*  layer_list,int layer_id);

#endif

