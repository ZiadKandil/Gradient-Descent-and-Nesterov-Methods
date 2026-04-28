.PHONY: clean

CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++23 -O2
INCLUDE_FLAGS = -I/usr/include/muparserx
LIB_FLAGS = -lmuparserx

main: main.o
	${CXX} ${CXXFLAGS} ${INCLUDE_FLAGS} -o main main.o ${LIB_FLAGS}

main.o : main.cpp
	${CXX} ${CXXFLAGS} ${INCLUDE_FLAGS} -c main.cpp

clean:
	rm -f main main.o