#/bin/bash
cudalib=/software/cuda-toolkit/9.2/lib64
# tf_inc=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
# tf_lib=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
tf_inc=/home/sniu/.conda/envs/tf/lib/python3.6/site-packages/tensorflow/include
tf_lib=/home/sniu/.conda/envs/tf/lib/python3.6/site-packages/tensorflow

INC=-I $(tf_inc) -I $(tf_inc)/external/nsync/public
LIB=-L $(tf_lib) -ltensorflow_framework

all: tf_approxmatch_so.so tf_approxmatch_g.cu.o tf_nndistance_so.so tf_nndistance_g.cu.o


tf_approxmatch_so.so: tf_approxmatch_g.cu.o tf_approxmatch.cpp
	g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC $(INC) -lcudart -L $(cudalib) $(LIB) -O2 -D_GLIBCXX_USE_CXX11_ABI=0


tf_approxmatch_g.cu.o: tf_approxmatch_g.cu
	nvcc -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_approxmatch_g.cu.o tf_approxmatch_g.cu $(INC) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2


tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
	g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC $(INC) -lcudart -L $(cudalib) $(LIB) -O2 -D_GLIBCXX_USE_CXX11_ABI=0


tf_nndistance_g.cu.o: tf_nndistance_g.cu
	nvcc -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu $(INC) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2


clean:
	rm tf_approxmatch_so.so
	rm tf_nndistance_so.so
	rm  *.cu.o 
