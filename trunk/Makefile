#Using nVidia common make for now.

# Set root dir to point to NVIDIA SDK
ROOTDIR    := /home/xmatela/NVIDIA_GPU_Computing_SDK/C/
ROOTBINDIR := ./bin


EXECUTABLE := dwt
CUFILES := main.cu dwt.cu components.cu 
CCFILES := 
CFILES := 

CFLAGS := -I ../ -std=c99
CXXFLAGS := -I ../

#USEGLUT=1
#USEGLLIB=1
verbose=1
dbg=1
OMIT_CUTIL_LIB=1

include $(ROOTDIR)/common/common.mk
