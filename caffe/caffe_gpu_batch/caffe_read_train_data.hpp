#ifndef	_CAFFE_READ_TRAIN_DATA_H__
#define	_CAFFE_READ_TRAIN_DATA_H__

#include <fstream>
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <map>
#include <vector>

using namespace std;


typedef struct region2d_box_{
	float x;
	float y;
	float w;
	float h;
    	float left, right, top, bottom;
}t_rgbox;


#define  	EMPTY_LABEL		-999
typedef	int	t_label;
typedef 	std::vector<std::pair<t_rgbox, int> > t_box_list;

typedef	std::vector<std::pair<std::string, t_box_list> > t_detection_train_data;

typedef	std::vector<std::pair<std::string, t_label> > t_classification_train_data;

typedef	std::vector<std::pair<int,std::pair<std::string, t_label> > > t_classification_train_data_load;

extern	int 	read_classification_train_data(char* filename, t_classification_train_data &td, int &loss_vect_dim);
extern	int 	read_single_image_data(string im_file,float* im_data_buf, int im_h, int im_w, int im_ch, float img_bias, float img_scale);
extern	int 	read_detection_train_data(char* filename, t_detection_train_data &td, int &loss_vect_dim);



#endif
