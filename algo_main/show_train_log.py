import matplotlib.pyplot as plt
import os, sys
import argparse
import scipy.io as sio
import numpy as np
import math


def show_log(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--flog",
		help="Input train log file."
		)
	parser.add_argument(
		"--title",
		default='',
		help="training title."
	)	
	args = parser.parse_args()
	args.flog = os.path.expanduser(args.flog)
	args.title = os.path.expanduser(args.title)	
	if not os.path.exists(args.flog):
		print("can not find the train log file: %s" % args.flog)

	log_file = open(args.flog, 'r') 
	iter_list=[]
	accuarcy_list=[]
	for line in log_file:
		list = line.split(',')
		iter_list.append(int(list[0]))
		accuarcy_list.append(float(list[2]))
	plt.plot(iter_list,accuarcy_list)
	y_axis_ticks=np.arange(0,1,0.04)
	plt.yticks(y_axis_ticks)
	plt.grid(True)
	plt.xlabel('training interation')
	plt.ylabel('test accuarcy')
	plt.title(args.title)
	plt.savefig(args.flog+".jpg")
	plt.show()
		
if __name__=='__main__':
    show_log(sys.argv)
