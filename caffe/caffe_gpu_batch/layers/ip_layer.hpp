#ifndef	_IP_LAYER_H
#define 	_IP_LAYER_H

#include<stdio.h>
#include<stdlib.h>
#include"../netsolve.hpp"

extern	int 	net_ip_forward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz);
extern	int 	net_ip_initial_para(t_layer_info*  layer_list,int layer_id,int in_ch);
extern	int 	net_ip_backward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz);

#endif
