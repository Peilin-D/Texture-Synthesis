import threading
import numpy as np


def func(arr, threadidx,num_threads):
	for i in range(threadidx,len(arr),num_threads):
		arr[i]=1;



if __name__=='__main__':

	array=np.zeros([1,100000])
	num_threads=4
	for threadidx in range(num_threads):
		th=threading.Thread(target=func,args=(array,threadidx, num_threads))
		th.start() 
	th.join()
	print array