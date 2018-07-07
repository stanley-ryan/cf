#ifndef		_LENET_TRAIN_GPU_H_
#define  	_LENET_TRAIN_GPU_H_

#include <vector>
#include <map>
#include<stdio.h>

typedef enum _n_lenet_train_layer_name{
	lenet_conv1=1,
	lenet_pool1,
	lenet_conv2,
	lenet_pool2,
	lenet_ip1,
	lenet_relu,
	lenet_ip2,
	lenet_softmax_loss,
}e_lenet_train_layer_name;





#endif
