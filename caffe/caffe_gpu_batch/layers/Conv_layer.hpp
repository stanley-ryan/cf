#ifndef	_CONV_LAYER_H
#define 	_CONV_LAYER_H

#include<stdio.h>
#include<stdlib.h>
#include"../netsolve.hpp"

extern	int 	net_conv_forward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz);
extern	int 	net_conv_initial_para(t_layer_info*  layer_list,int layer_id,int in_ch);
extern	int 	net_conv_backward_batch(t_layer_info*  layer_list,int layer_id,int batch_sz);
extern	void  im2col_ongpu(float *im,  int channels, int height, int width,  int ksize, int stride, int pad, float *data_col);
extern	int 	cuda_add_test(int N, float* x, float* y);
#endif
