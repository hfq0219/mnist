#请确保电脑上已安装配置好 opencv3 开发环境

CPP=g++
EXEC=run
OBJDIR=./obj/

LDFLAGS+= `pkg-config --libs opencv` -std=c++11
COMMON+= `pkg-config --cflags opencv`

OBJ = main.o layer.o network.o
OBJS = $(addprefix $(OBJDIR), $(OBJ))

all: obj data $(EXEC)

$(EXEC): $(OBJS)
	$(CPP) $^ -o $@ $(COMMON) $(LDFLAGS) 

$(OBJDIR)%.o: %.cpp
	$(CPP) -c $< -o $@ $(COMMON) $(LDFLAGS)

data:mnist.cpp dic
	$(CPP) $< -o $@ $(COMMON) $(LDFLAGS)

obj:
	mkdir -p obj

dic:
	mkdir -p testData trainData backup

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC) data