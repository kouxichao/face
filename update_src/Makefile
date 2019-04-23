debug =
MODE = 
RGB_FORMAT =
CFLAGS += -I$(PWD) \
	-I$(PWD)/../ncnn \
	-I$(PWD)/../sqlite \
	-I$(PWD)/../dlib \
	-L$(PWD)/../lib -lsqlite3 -ldlib -lncnn -ldl
CFLAGS += $(MODE) $(debug) $(RGB_FORMAT) -O3 -fopenmp -mcpu=cortex-a53+simd -mcpu=cortex-a53+fp -fpermissive 
#SRCS := $(wildcard *.cpp)
SRCS := interface_face.cpp 

ifeq ($(MODE), -DJPG_DEMO)
all:detect_face interface_face evaluate demo_face 
else 
all:interface_face demo_face
endif

detect_face: 
	aarch64-himix100-linux-g++ -o detect_face detect_face.cpp $(CFLAGS) 

interface_face:
	aarch64-himix100-linux-g++ $(SRCS) $(CFLAGS) -c -o interface_face.cpp.o 
	ar r libface.a interface_face.cpp.o

interface_face_switchalign:
	aarch64-himix100-linux-g++ interface_face_switchalign.cpp $(CFLAGS) -c -o interface_face.cpp.o 
	ar r libface.a interface_face.cpp.o

demo_face_switchalign:interface_face_switchalign
	aarch64-himix100-linux-g++ demo_face.cpp  libface.a $(CFLAGS) -o demo_face 

demo_face:interface_face
	aarch64-himix100-linux-g++ demo_face.cpp  libface.a $(CFLAGS) -o demo_face 

evaluate:interface_face
	aarch64-himix100-linux-g++  evaluate.cpp    libface.a $(CFLAGS) -o evaluate 
