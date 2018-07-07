#ifndef		_CAFFE_TRAIN_CLASSFICATION_H_
#define 	_CAFFE_TRAIN_CLASSFICATION_H_


#include"netsolve.hpp"


typedef struct  _loss_info{
	int layer_id;
	float weight;
}t_loss_info;


#if (TRAIN_MODE==1)			//classification mode
extern	int  caffe_train_classfication_withload_iTest(char* train_file, char* test_file, t_layer_info* net,  int im_ch,int im_h,int im_w,float img_bias, float img_scale, t_train_strat* train_config, char* train_tag);
#endif
#endif
