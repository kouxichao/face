CFLAGS += -I$(PWD) \
	-I$(PWD)/../ncnn \
	-I$(PWD)/../sqlite \
	-I$(PWD)/../dlib \
	-L$(PWD)/../lib -lsqlite3 -ldlib -lncnn -ldl
CFLAGS += -O3 -fopenmp -mcpu=cortex-a53+simd -mcpu=cortex-a53+fp 
#SRCS := $(wildcard *.cpp)
SRCS := interface_face.cpp 

all:interface_face demo_face

interface_face:
	aarch64-himix100-linux-g++ $(SRCS) $(CFLAGS) -c -o interface_face.cpp.o 
	ar r libface.a interface_face.cpp.o

demo_face:interface_face
	aarch64-himix100-linux-g++ demo_face.cpp  libface.a $(CFLAGS) -o demo_face 
