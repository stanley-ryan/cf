#include <iostream>
#include<string.h>
#include <stdio.h>
#include <getopt.h>
#include <sys/time.h>
#include"netsolve.hpp"
#include"caffe_train_classification.hpp"
#include <opencv2/opencv.hpp>
#include "lenet_train.hpp"
#include"random_fill.hpp"
using namespace std;
using namespace cv;


#define EPS		0.0001



float lenet_conv1_w[20][1][5][5];
float lenet_conv1_b[1][1][1][20];
float lenet_conv2_w[50][20][5][5];
float lenet_conv2_b[1][1][1][50];
float lenet_ip1_w[1][1][500][800];
float lenet_ip1_b[1][1][1][500];
float lenet_ip2_w[1][1][10][500];
float lenet_ip2_b[1][1][1][10];
float lenet_relu_slope[1][1][1][1]={0.0};
float lenet_softmax_loss_weight[1][1][1][1]={1.0};


t_layer_info  lenet_train_net[ ]={
{lenet_conv1,			layer_conv,		20,		{-1,-1},			{5,1,0,1,act_none},	(float*)lenet_conv1_w,	(float*)lenet_conv1_b,	},
{lenet_pool1,			layer_pool, 		20, 		{lenet_conv1,-1},	{2,2,0,0,pool_max}, 	NULL,				NULL,				},
{lenet_conv2,			layer_conv,		50,		{lenet_pool1,-1},	{5,1,0,1,act_none},	(float*)lenet_conv2_w,	(float*)lenet_conv2_b,	},
{lenet_pool2,			layer_pool, 		50, 		{lenet_conv2,-1},	{2,2,0,0,pool_max}, 	NULL,				NULL,				},
{lenet_ip1,			layer_ip,			500,		{lenet_pool2,-1},	{0,0,0,0,act_none},	(float*)lenet_ip1_w,		(float*)lenet_ip1_b,		},
{lenet_relu,			layer_Relu,		500,		{lenet_ip1,-1},		{0,0,0,0,act_none},	(float*)lenet_relu_slope,	NULL,				},
{lenet_ip2,			layer_ip, 			10, 		{lenet_relu,-1},		{0,0,0,0,act_none}, 	(float*)lenet_ip2_w,		(float*)lenet_ip2_b,		},
{lenet_softmax_loss,	layer_Softmax_loss,	10,		{lenet_ip2,-1},		{0,0,0,0,act_none},	NULL,				(float*)lenet_softmax_loss_weight,	},
{-1,					layer_none, 		0,		{0},				{0,0,0,0,act_none}, 	NULL,				NULL,				},
};



int main( int argc, char** argv )
{
	t_train_strat train_config;
	train_config.lr_mode=lr_inv;
	train_config.base_lr=0.008;
	train_config.gamma=0.0001;
	train_config.setp=1;
	train_config.batch_size=2;
	train_config.power=0.75;
	train_config.iter_size=32;
	train_config.test_size=20;
	train_config.max_iter=9999999999;
	train_config.moment=0.9;
	train_config.target_accuracy=0.986;
	train_config.decay=0.0005;
	train_config.loss_layer_num=1;
	train_config.loss_layer[0]=7;
	train_config.is_shuffle=1;
	train_config.is_withload=1;
	train_config.record_itersize=20;

	lenet_train_net[0].train_inf.para1_lrmul=1;
	lenet_train_net[0].train_inf.para1_ini_way=para_ini_xavier;
	lenet_train_net[0].train_inf.para2_lrmul=2;
	lenet_train_net[0].train_inf.para2_ini_way=para_ini_const_zero;
	lenet_train_net[2].train_inf.para1_lrmul=1;
	lenet_train_net[2].train_inf.para1_ini_way=para_ini_xavier;
	lenet_train_net[2].train_inf.para2_lrmul=2;
	lenet_train_net[2].train_inf.para2_ini_way=para_ini_const_zero;
	lenet_train_net[4].train_inf.para1_lrmul=1;
	lenet_train_net[4].train_inf.para1_ini_way=para_ini_xavier;
	lenet_train_net[4].train_inf.para2_lrmul=2;
	lenet_train_net[4].train_inf.para2_ini_way=para_ini_const_zero;
	lenet_train_net[6].train_inf.para1_lrmul=1;
	lenet_train_net[6].train_inf.para1_ini_way=para_ini_xavier;
	lenet_train_net[6].train_inf.para2_lrmul=2;
	lenet_train_net[6].train_inf.para2_ini_way=para_ini_const_zero;


#if 0
	lenet_train_net[0].train_inf.para_diff_way=e_para1_prune;
	lenet_train_net[0].train_inf.prune_percent=0.9;
	lenet_train_net[2].train_inf.para_diff_way=e_para1_prune;
	lenet_train_net[2].train_inf.prune_percent=0.9;
	lenet_train_net[4].train_inf.para_diff_way=e_para1_prune;
	lenet_train_net[4].train_inf.prune_percent=0.9;
	lenet_train_net[6].train_inf.para_diff_way=e_para1_prune;
	lenet_train_net[6].train_inf.prune_percent=0.9;
#endif
	printf("#############start the lenet full weights train\n");
	caffe_train_classfication_withload_iTest("/media/fish/DATA/dev/mnist_im/train_file.txt", 
				"/media/fish/DATA/dev/mnist_im/test_file.txt", 
				lenet_train_net,1,28,28,0, 0.00390625, &train_config,
				"lenet_prune90_train");

}
